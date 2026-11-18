from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tgmr_det.config import load_config
from tgmr_det.data import PairedVisibleThermalDataset, paired_collate_fn
from tgmr_det.engine import train_one_epoch
from tgmr_det.losses import DetectionLoss
from tgmr_det.factory import build_model
from tgmr_det.utils import (
    evaluate_ap50,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)


def build_dataset(cfg, split, training):
    d, t = cfg["data"], cfg["train"]

    return PairedVisibleThermalDataset(
        root=d["root"],
        split=split,
        image_size=t["image_size"],
        visible_dir=d["visible_dir"],
        thermal_dir=d["thermal_dir"],
        labels_dir=d["labels_dir"],
        image_extensions=d["image_extensions"],
        training=training,
        horizontal_flip=t["horizontal_flip"] if training else 0.0,
        visible_color_jitter=t["visible_color_jitter"] if training else 0.0,
        thermal_contrast_jitter=t["thermal_contrast_jitter"] if training else 0.0,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aeromis_vt.yaml")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(int(cfg.get("seed", 42)))

    requested = args.device or cfg.get("device", "cuda")
    device = torch.device(
        requested if requested == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_set = build_dataset(cfg, cfg["data"]["train_split"], True)
    val_set = build_dataset(cfg, cfg["data"]["val_split"], False)

    loader_common = dict(
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        collate_fn=paired_collate_fn,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        **loader_common,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=max(1, cfg["train"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    model = build_model(cfg).to(device)
    m, t = cfg["model"], cfg["train"]

    print(
        "Model settings: "
        f"use_p2_head={model.use_p2_head}, "
        f"active_strides={model.strides}, "
        f"head_variant={model.head_variant}"
    )

    # Full TGMR-Det uses P2--P5. The controlled no-P2 ablation predicts only
    # at P3--P5, so loss assignment must use active model strides and matching
    # scale ranges rather than the four-level YAML defaults.
    regression_ranges = (
        m["regression_ranges"]
        if model.use_p2_head
        else m["regression_ranges"][1:]
    )

    if len(regression_ranges) != len(model.strides):
        raise RuntimeError(
            f"Mismatch between regression ranges ({len(regression_ranges)}) "
            f"and active strides ({len(model.strides)})."
        )

    criterion = DetectionLoss(
        num_classes=m["num_classes"],
        strides=model.strides,
        regression_ranges=regression_ranges,
        center_sampling_radius=m["center_sampling_radius"],
        cls_weight=t["cls_weight"],
        ctr_weight=t["ctr_weight"],
        box_weight=t["box_weight"],
        giou_weight=t["giou_weight"],
        nwd_weight=t["nwd_weight"],
        nwd_constant=t["nwd_constant"],
        align_weight=t.get("align_weight", 0.02),
        flow_smooth_weight=t.get("flow_smooth_weight", 1.0),
        flow_magnitude_weight=t.get("flow_magnitude_weight", 0.05),
        affine_weight=t.get("affine_weight", 0.02),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t["lr"],
        weight_decay=t["weight_decay"],
    )

    warmup = max(1, int(t["warmup_epochs"]))
    epochs = int(t["epochs"])

    def lr_lambda(epoch):
        if epoch < warmup:
            return max(0.1, (epoch + 1) / warmup)

        progress = (epoch - warmup) / max(1, epochs - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        return t["min_lr_ratio"] + (1 - t["min_lr_ratio"]) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=t["amp"] and device.type == "cuda",
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=t["amp"] and device.type == "cuda",
        )

    output_dir = Path(t["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_ap = 0.0

    if t.get("resume"):
        checkpoint = load_checkpoint(
            t["resume"],
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_ap = float(checkpoint.get("best_ap", 0.0))

        print(
            f"Resumed from epoch {start_epoch}, "
            f"best AP50={best_ap:.4f}"
        )

    for epoch in range(start_epoch, epochs):
        stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            amp=t["amp"],
            accumulate=t["accumulate"],
            grad_clip=t["grad_clip"],
        )

        scheduler.step()
        print(f"Epoch {epoch}: {stats}")

        if (epoch + 1) % int(t["eval_every"]) == 0:
            metrics = evaluate_ap50(
                model,
                val_loader,
                device,
                conf_threshold=cfg["eval"]["conf_threshold"],
                nms_iou_threshold=cfg["eval"]["nms_iou_threshold"],
                max_detections=cfg["eval"]["max_detections"],
            )

            ap = metrics["map50"]
            print(f"Validation: {metrics}")

            if ap >= best_ap:
                best_ap = ap

                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_ap,
                    cfg,
                )

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_ap,
            cfg,
        )

        if (epoch + 1) % int(t["save_every"]) == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_ap,
                cfg,
            )


if __name__ == "__main__":
    main()