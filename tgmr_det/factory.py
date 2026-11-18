"""Single entrypoint to avoid train/eval/infer architecture drift."""
from .model import TGMRDet


def build_model(cfg):
    m = cfg['model']
    return TGMRDet(
        num_classes=m['num_classes'],
        width=m.get('width', 0.75),
        neck_channels=m['neck_channels'],
        itsm_dim=m['itsm_dim'],
        itsm_heads=m['itsm_heads'],
        itsm_pool=m['itsm_pool'],
        max_shift=m['max_shift'],
        max_log_scale=m['max_log_scale'],
        max_local_shift=m.get('max_local_shift', 2.0),
        sparse_attention=m.get('sparse_attention', True),
        sparse_group_size=m.get('sparse_group_size', 2),
        use_itsm=m.get('use_itsm', True),
        use_msfr=m.get('use_msfr', True),
        use_ssfcfm=m.get('use_ssfcfm', True),
        backbone_type=m.get('backbone_type', 'official_yolo11n'),
        pretrained_yolo11=m.get('pretrained_yolo11'),
        strides=m.get('strides', (4, 8, 16, 32)),
        use_p2_head=m.get('use_p2_head', True),
        head_variant=m.get('head_variant', 'decoupled'),
    )
