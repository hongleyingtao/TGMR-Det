from __future__ import annotations

from torch import nn


class OfficialYOLO11nBackbone(nn.Module):
    """Exact first 11 backbone modules from Ultralytics YOLO11n detection.

    Features are tapped at stages 2,4,6,10 (P2,P3,P4,P5). The Ultralytics
    Detect head and pretrained YOLO11 neck are NOT used, consistent with the
    TGMR-Det paper's custom PAN-FPN and FCOS-like prediction head.
    """

    def __init__(self, weights: str | None = None):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Official YOLO11n backbone requires 'ultralytics'; run pip install -r requirements.txt. "
                "Offline smoke tests can use configs/demo.yaml with csp_demo."
            ) from exc
        source = str(weights) if weights else 'yolo11n.yaml'
        model = YOLO(source).model
        backbone_modules = list(model.model.children())[:11]
        if len(backbone_modules) != 11:
            raise RuntimeError('Unexpected YOLO11n backbone. Pin/check ultralytics version.')
        self.layers = nn.ModuleList(backbone_modules)
        # YOLO11n width multiplier 0.25; channels after stages 2,4,6,10.
        self.channels = (64, 128, 128, 256)

    def forward(self, x):
        results = []
        for index, layer in enumerate(self.layers):
            if getattr(layer, 'f', -1) != -1:
                raise RuntimeError('Unexpected backbone skip connection in Ultralytics YOLO11n')
            x = layer(x)
            if index in (2, 4, 6, 10):
                results.append(x)
        for expected, feat in zip(self.channels, results):
            if feat.shape[1] != expected:
                raise RuntimeError(f'YOLO11n channel mismatch: expected {expected}, got {feat.shape[1]}')
        return results
