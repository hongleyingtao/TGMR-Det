import argparse
import torch
from torch.utils.data import DataLoader

from tgmr_det.config import load_config
from tgmr_det.data import PairedVisibleThermalDataset, paired_collate_fn
from tgmr_det.factory import build_model
from tgmr_det.utils import evaluate_ap50, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/aeromis_vt.yaml")
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    m, d, t = cfg["model"], cfg["data"], cfg["train"]

    cfg['model']['pretrained_yolo11'] = None
    model = build_model(cfg).to(device)
    load_checkpoint(args.weights, model, map_location=device)

    dataset = PairedVisibleThermalDataset(
        root=d["root"], split=d["val_split"], image_size=t["image_size"],
        visible_dir=d["visible_dir"], thermal_dir=d["thermal_dir"],
        labels_dir=d["labels_dir"], image_extensions=d["image_extensions"],
        training=False,
    )
    loader = DataLoader(
        dataset, batch_size=t["batch_size"], shuffle=False,
        num_workers=cfg.get("num_workers", 4), pin_memory=device.type == "cuda",
        collate_fn=paired_collate_fn,
    )
    print(evaluate_ap50(
        model, loader, device,
        conf_threshold=cfg["eval"]["conf_threshold"],
        nms_iou_threshold=cfg["eval"]["nms_iou_threshold"],
        max_detections=cfg["eval"]["max_detections"],
    ))


if __name__ == "__main__":
    main()
