import argparse
from pathlib import Path

import torch
from PIL import Image

from tgmr_det.config import load_config
from tgmr_det.data import letterbox, pil_to_tensor
from tgmr_det.factory import build_model
from tgmr_det.utils import draw_detections, load_checkpoint, reverse_letterbox_boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/aeromis_vt.yaml")
    p.add_argument("--weights", required=True)
    p.add_argument("--visible", required=True)
    p.add_argument("--thermal", required=True)
    p.add_argument("--output", default="prediction_thermal.jpg")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    m, t = cfg["model"], cfg["train"]

    cfg['model']['pretrained_yolo11'] = None
    model = build_model(cfg).to(device)
    load_checkpoint(args.weights, model, map_location=device)
    model.eval()

    visible_original = Image.open(args.visible).convert("RGB")
    thermal_original = Image.open(args.thermal).convert("RGB")
    visible, _, _ = letterbox(visible_original, t["image_size"])
    thermal, thermal_ratio, thermal_pad = letterbox(thermal_original, t["image_size"])

    with torch.no_grad():
        result = model.decode(
            model(
                pil_to_tensor(visible).unsqueeze(0).to(device),
                pil_to_tensor(thermal).unsqueeze(0).to(device),
            ),
            conf_threshold=cfg["infer"]["conf_threshold"],
            nms_iou_threshold=cfg["infer"]["nms_iou_threshold"],
            max_detections=cfg["infer"]["max_detections"],
        )[0]

    boxes = reverse_letterbox_boxes(
        result["boxes"].cpu(), thermal_ratio, thermal_pad,
        (thermal_original.height, thermal_original.width),
    )
    rendered = draw_detections(
        thermal_original, boxes, result["scores"].cpu(), result["labels"].cpu(),
        cfg["data"]["class_names"],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rendered.save(args.output)
    print(f"Saved {len(boxes)} detections to {args.output}")
    print("Output boxes use the thermal reference coordinate system.")


if __name__ == "__main__":
    main()
