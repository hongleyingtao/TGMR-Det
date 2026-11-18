import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from tgmr_det.config import load_config
from tgmr_det.data import _find_image, load_yolo_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/aeromis_vt.yaml")
    p.add_argument("--split", default="train")
    args = p.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    root = Path(d["root"]) / args.split
    thermal_dir = root / d["thermal_dir"]
    visible_dir = root / d["visible_dir"]
    labels_dir = root / d["labels_dir"]
    exts = tuple(d["image_extensions"])

    thermal_files = [
        p for p in sorted(thermal_dir.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    missing_visible, missing_labels, bad_labels = [], [], []
    box_sizes, class_counts = [], Counter()

    for thermal_path in thermal_files:
        stem = thermal_path.stem
        try:
            _find_image(visible_dir, stem, exts)
        except FileNotFoundError:
            missing_visible.append(stem)

        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            missing_labels.append(stem)
            continue

        try:
            size = Image.open(thermal_path).size
            boxes, labels = load_yolo_labels(label_path, size)
            for box, label in zip(boxes, labels):
                box_sizes.append((float(box[2] - box[0]), float(box[3] - box[1])))
                class_counts[int(label)] += 1
        except Exception as exc:
            bad_labels.append((stem, str(exc)))

    total = len(box_sizes)
    tiny16 = sum(w < 16 and h < 16 for w, h in box_sizes)
    tiny32 = sum(w < 32 and h < 32 for w, h in box_sizes)
    print(f"Images: {len(thermal_files)}")
    print(f"Objects: {total}")
    print(f"Class counts: {dict(class_counts)}")
    print(f"Both width and height <16 px: {tiny16}/{total} ({tiny16/max(total,1):.1%})")
    print(f"Both width and height <32 px: {tiny32}/{total} ({tiny32/max(total,1):.1%})")
    print(f"Missing visible: {len(missing_visible)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Bad labels: {len(bad_labels)}")
    if missing_visible[:10]:
        print("Missing visible examples:", missing_visible[:10])
    if bad_labels[:10]:
        print("Bad label examples:", bad_labels[:10])


if __name__ == "__main__":
    main()
