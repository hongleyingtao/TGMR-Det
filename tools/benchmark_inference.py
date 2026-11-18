"""Report measured FPS under an explicit local hardware protocol.

Do not compare with the paper's 72.56 FPS unless resolution, GPU, model,
precision, CUDA setup, NMS handling, and warmup are identical.
"""
import argparse, sys, time, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from tgmr_det.config import load_config
from tgmr_det.factory import build_model
from tgmr_det.utils import load_checkpoint

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/aeromis_vt.yaml')
    p.add_argument('--weights',default=None)
    p.add_argument('--device',default='cuda')
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--size',type=int,default=None)
    args=p.parse_args()
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('No CUDA GPU detected')
    cfg=load_config(args.config)
    cfg['model']['pretrained_yolo11']=None
    model=build_model(cfg).to(device).eval()
    if args.weights:load_checkpoint(args.weights,model,map_location=device)
    size=args.size or cfg['train']['image_size']
    vis=torch.rand((1,3,size,size),device=device)
    ir=torch.rand((1,3,size,size),device=device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    for _ in range(args.warmup): model(vis,ir)
    sync()
    t0=time.perf_counter()
    for _ in range(args.repeats): model(vis,ir)
    sync()
    elapsed=time.perf_counter()-t0
    print(json.dumps({'fps_forward_only':args.repeats/elapsed,'ms_per_pair_forward_only':1000*elapsed/args.repeats,
        'parameters_million':sum(p.numel() for p in model.parameters())/1e6,
        'input_size':size,'device':str(device),'torch':torch.__version__,
        'checkpoint_loaded':bool(args.weights),'notes':'Excludes data loading and NMS. Do not equate with paper FPS without protocol match.'},indent=2))
if __name__=='__main__':main()
