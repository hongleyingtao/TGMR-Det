"""Generate explicit ablation YAMLs; does not fabricate experimental metrics."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tgmr_det.config import load_config,save_config

MODULES=('itsm','msfr','ssfcfm')
VARIANTS={
    'full':{},
    'baseline':{'use_itsm':False,'use_msfr':False,'use_ssfcfm':False},
    'itsm':{'use_itsm':True,'use_msfr':False,'use_ssfcfm':False},
    'msfr':{'use_itsm':False,'use_msfr':True,'use_ssfcfm':False},
    'ssfcfm':{'use_itsm':False,'use_msfr':False,'use_ssfcfm':True},
    'itsm_msfr':{'use_itsm':True,'use_msfr':True,'use_ssfcfm':False},
    'itsm_ssfcfm':{'use_itsm':True,'use_msfr':False,'use_ssfcfm':True},
    'msfr_ssfcfm':{'use_itsm':False,'use_msfr':True,'use_ssfcfm':True},
    'no_s3a':{'sparse_attention':False},
    'no_p2':{'use_p2_head':False},
    'shared_head':{'head_variant':'shared'},
    'no_nwd':{'nwd_weight':0.0,'giou_weight':1.0},
    'no_alignment_loss':{'align_weight':0.0},
}

def generate(base,variant,out):
    cfg=load_config(base)
    for key,value in VARIANTS[variant].items():
        group='train' if key in ('nwd_weight','giou_weight','align_weight') else 'model'
        cfg[group][key]=value
    cfg['train']['output_dir']=str(Path('runs')/'ablations'/variant)
    save_config(cfg,out)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/aeromis_vt.yaml')
    p.add_argument('--variant',choices=tuple(VARIANTS),required=True)
    p.add_argument('--output',default=None)
    args=p.parse_args()
    output=args.output or str(Path('runs')/'ablation_configs'/(args.variant+'.yaml'))
    print('Wrote',generate(args.config,args.variant,output))
    print('Ablations are method options, not reproduced values from the paper tables.')
