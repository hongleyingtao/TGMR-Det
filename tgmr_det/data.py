from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


def _find_image(directory: Path, stem: str, extensions: Sequence[str]) -> Path:
    for ext in extensions:
        for candidate in (directory / f"{stem}{ext}", directory / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Cannot find image '{stem}' in {directory}")


def letterbox(
    image: Image.Image,
    size: int,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """Resize with unchanged aspect ratio and pad to size x size."""
    w, h = image.size
    ratio = min(size / w, size / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas, ratio, (left, top)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_yolo_labels(path: Path, image_size: Tuple[int, int]):
    """Read class xc yc w h in normalized ORIGINAL THERMAL coordinates."""
    width, height = image_size
    boxes: List[List[float]] = []
    labels: List[int] = []
    if not path.exists():
        raise FileNotFoundError(
            f"Missing thermal label file: {path}. For a true background or visible-only image, "
            "create an explicit EMPTY thermal label .txt file."
        )

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 values, got {len(parts)}")
        cls, xc, yc, bw, bh = map(float, parts)
        x1 = max(0.0, (xc - bw / 2.0) * width)
        y1 = max(0.0, (yc - bh / 2.0) * height)
        x2 = min(float(width), (xc + bw / 2.0) * width)
        y2 = min(float(height), (yc + bh / 2.0) * height)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls))

    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.long),
    )


class PairedVisibleThermalDataset(Dataset):
    """
    Layout:
      root/train/visible, root/train/thermal, root/train/labels
      root/val/visible,   root/val/thermal,   root/val/labels

    Labels are YOLO txt files in the ORIGINAL THERMAL coordinate system.
    Visible and thermal files must share the same stem.
    """

    def __init__(
        self,
        root,
        split: str,
        image_size: int,
        visible_dir: str = "visible",
        thermal_dir: str = "thermal",
        labels_dir: str = "labels",
        image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
        training: bool = False,
        horizontal_flip: float = 0.0,
        visible_color_jitter: float = 0.0,
        thermal_contrast_jitter: float = 0.0,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.visible_root = self.root / split / visible_dir
        self.thermal_root = self.root / split / thermal_dir
        self.labels_root = self.root / split / labels_dir
        self.extensions = tuple(x.lower() for x in image_extensions)
        self.training = training
        self.horizontal_flip = float(horizontal_flip)
        self.visible_color_jitter = float(visible_color_jitter)
        self.thermal_contrast_jitter = float(thermal_contrast_jitter)

        if not self.thermal_root.is_dir():
            raise FileNotFoundError(f"Thermal directory not found: {self.thermal_root}")
        if not self.visible_root.is_dir():
            raise FileNotFoundError(f"Visible directory not found: {self.visible_root}")

        thermal_files = [
            p for p in sorted(self.thermal_root.iterdir())
            if p.is_file() and p.suffix.lower() in self.extensions
        ]
        if not thermal_files:
            raise RuntimeError(f"No thermal images found in {self.thermal_root}")

        self.samples = []
        for thermal_path in thermal_files:
            stem = thermal_path.stem
            visible_path = _find_image(self.visible_root, stem, self.extensions)
            self.samples.append(
                (visible_path, thermal_path, self.labels_root / f"{stem}.txt", stem)
            )

    def __len__(self):
        return len(self.samples)

    def _jitter_visible(self, image):
        s = self.visible_color_jitter
        if s <= 0:
            return image
        image = ImageEnhance.Brightness(image).enhance(random.uniform(1 - s, 1 + s))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(1 - s, 1 + s))
        return ImageEnhance.Color(image).enhance(random.uniform(1 - s, 1 + s))

    def _jitter_thermal(self, image):
        s = self.thermal_contrast_jitter
        if s <= 0:
            return image
        image = ImageEnhance.Contrast(image).enhance(random.uniform(1 - s, 1 + s))
        return ImageEnhance.Brightness(image).enhance(random.uniform(1 - s, 1 + s))

    def __getitem__(self, index):
        visible_path, thermal_path, label_path, stem = self.samples[index]
        visible_original = Image.open(visible_path).convert("RGB")
        thermal_original = Image.open(thermal_path).convert("RGB")

        boxes, labels = load_yolo_labels(label_path, thermal_original.size)
        thermal_size = thermal_original.size
        visible_size = visible_original.size

        visible, visible_ratio, visible_pad = letterbox(visible_original, self.image_size)
        thermal, thermal_ratio, thermal_pad = letterbox(thermal_original, self.image_size)

        if boxes.numel() > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * thermal_ratio + thermal_pad[0]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * thermal_ratio + thermal_pad[1]
            boxes.clamp_(0, self.image_size)

        if self.training:
            visible = self._jitter_visible(visible)
            thermal = self._jitter_thermal(thermal)
            if random.random() < self.horizontal_flip:
                visible = visible.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                thermal = thermal.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if boxes.numel() > 0:
                    x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
                    boxes[:, 0] = self.image_size - x2
                    boxes[:, 2] = self.image_size - x1

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.long),
            "stem": stem,
            "thermal_original_size": torch.tensor([thermal_size[1], thermal_size[0]]),
            "visible_original_size": torch.tensor([visible_size[1], visible_size[0]]),
            "thermal_ratio": torch.tensor(thermal_ratio, dtype=torch.float32),
            "thermal_pad": torch.tensor(thermal_pad, dtype=torch.float32),
            "visible_ratio": torch.tensor(visible_ratio, dtype=torch.float32),
            "visible_pad": torch.tensor(visible_pad, dtype=torch.float32),
            "thermal_path": str(thermal_path),
            "visible_path": str(visible_path),
        }
        return pil_to_tensor(visible), pil_to_tensor(thermal), target


def paired_collate_fn(batch):
    return (
        torch.stack([x[0] for x in batch], dim=0),
        torch.stack([x[1] for x in batch], dim=0),
        [x[2] for x in batch],
    )
