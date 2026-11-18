import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tgmr_det.config import load_config
from tgmr_det.factory import build_model
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/aeromis_vt.yaml');a=p.parse_args()
cfg=load_config(a.config);cfg['model']['pretrained_yolo11']=None
m=build_model(cfg)
print(f"{sum(p.numel() for p in m.parameters())/1e6:.4f} M parameters; this is not a verified reproduction of the paper's 13.86 M.")
