from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_targets_to_device(targets, device):
    moved = []
    for target in targets:
        moved.append({
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in target.items()
        })
    return moved


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_ap, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_ap": best_ap,
        "config": cfg,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def reverse_letterbox_boxes(boxes, ratio, pad, original_size):
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - float(pad[0])) / float(ratio)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - float(pad[1])) / float(ratio)
    h, w = int(original_size[0]), int(original_size[1])
    boxes[:, [0, 2]].clamp_(0, w)
    boxes[:, [1, 3]].clamp_(0, h)
    return boxes


def draw_detections(image, boxes, scores, labels, class_names: Sequence[str], line_width=3):
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, image.width // 80))
    except OSError:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        x1, y1, x2, y2 = box
        name = class_names[label] if 0 <= label < len(class_names) else str(label)
        text = f"{name} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)
        tb = draw.textbbox((x1, y1), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=(255, 0, 0))
        draw.text((x1 + 2, ty + 2), text, fill=(255, 255, 255), font=font)
    return image


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))



@torch.no_grad()
def evaluate_ap50(
    model, loader, device,
    conf_threshold=0.05, nms_iou_threshold=0.5, max_detections=300,
):
    model.eval()
    predictions, ground_truths = [], []

    for visible, thermal, targets in loader:
        visible = visible.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        results = model.decode(
            model(visible, thermal),
            conf_threshold, nms_iou_threshold, max_detections,
        )
        for result, target in zip(results, targets):
            predictions.append({
                "boxes": result["boxes"].detach().cpu(),
                "scores": result["scores"].detach().cpu(),
                "labels": result["labels"].detach().cpu(),
            })
            ground_truths.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })

    aps, per_class = [], {}
    for cls in range(model.num_classes):
        cls_preds, total_gt, gt_used = [], 0, {}
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            p_mask = pred["labels"] == cls
            g_mask = gt["labels"] == cls
            p_boxes, p_scores = pred["boxes"][p_mask], pred["scores"][p_mask]
            g_boxes = gt["boxes"][g_mask]
            total_gt += len(g_boxes)
            gt_used[img_idx] = torch.zeros(len(g_boxes), dtype=torch.bool)
            for box, score in zip(p_boxes, p_scores):
                cls_preds.append((float(score), img_idx, box))

        if total_gt == 0:
            per_class[cls] = float("nan")
            continue

        if not cls_preds:
            aps.append(0.0)
            per_class[cls] = 0.0
            continue

        cls_preds.sort(key=lambda x: x[0], reverse=True)
        tp = np.zeros(len(cls_preds), dtype=np.float32)
        fp = np.zeros(len(cls_preds), dtype=np.float32)

        for i, (_, img_idx, box) in enumerate(cls_preds):
            gt = ground_truths[img_idx]
            g_boxes = gt["boxes"][gt["labels"] == cls]
            if len(g_boxes) == 0:
                fp[i] = 1
                continue
            ious = box_iou(box.unsqueeze(0), g_boxes).squeeze(0)
            best_iou, best_idx = ious.max(dim=0)
            if best_iou >= 0.5 and not gt_used[img_idx][best_idx]:
                tp[i] = 1
                gt_used[img_idx][best_idx] = True
            else:
                fp[i] = 1

        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recall = tp / max(total_gt, 1)
        precision = tp / np.maximum(tp + fp, 1e-12)
        ap = compute_ap(recall, precision)
        aps.append(ap)
        per_class[cls] = ap

    return {
        "map50": float(np.mean(aps)) if aps else 0.0,
        "per_class_ap50": per_class,
    }
