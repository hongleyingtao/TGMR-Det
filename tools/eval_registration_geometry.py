"""Paired-box geometry check: use ONE unambiguous co-visible UAV pair/image.

The sampling field maps thermal grid coordinates to locations in the visible
feature. Evaluation uses 5 point samples per box, an approximate bounding box.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tgmr_det.config import load_config
from tgmr_det.data import PairedVisibleThermalDataset, paired_collate_fn, load_yolo_labels
from tgmr_det.factory import build_model
from tgmr_det.utils import load_checkpoint, reverse_letterbox_boxes
from tgmr_det.geometry import normalized_box,center_displacement,scale_ratio,alignment_iou,map_thermal_box_to_visible

def summarize(rows):
    if not rows:return None
    x=np.array(rows,dtype=float)
    return {'N':len(rows),'center_med':float(np.median(x[:,0])),
        'center_p90':float(np.percentile(x[:,0],90)),
        'scale_ratio_med':float(np.median(x[:,1])),
        'alignment_iou_mean':float(x[:,2].mean())}

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/aeromis_vt.yaml')
    p.add_argument('--weights',required=True)
    p.add_argument('--split',default='test')
    p.add_argument('--data-root',default=None)
    p.add_argument('--device',default='cpu')
    args=p.parse_args(); cfg=load_config(args.config)
    if args.data_root:cfg['data']['root']=args.data_root
    d,t=cfg['data'],cfg['train']; cfg['model']['pretrained_yolo11']=None
    model=build_model(cfg).to(args.device).eval()
    load_checkpoint(args.weights,model,map_location=args.device)
    ds=PairedVisibleThermalDataset(d['root'],args.split,t['image_size'],d['visible_dir'],d['thermal_dir'],d['labels_dir'],d['image_extensions'])
    dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=cfg.get('num_workers',0),collate_fn=paired_collate_fn)
    rows={'before':[], **{f'full_p{i}':[] for i in range(2,6)}}
    n_skip=0
    for v,ir,targets in dl:
        q=targets[0]; stem=q['stem']
        vh,vw=q['visible_original_size'].tolist(); th,tw=q['thermal_original_size'].tolist()
        vp=Path(d['root'])/args.split/d.get('visible_labels_dir','visible_labels')/(stem+'.txt')
        if not vp.is_file():n_skip+=1;continue
        v_boxes,_=load_yolo_labels(vp,(vw,vh));t_boxes,_=load_yolo_labels(Path(d['root'])/args.split/d['labels_dir']/(stem+'.txt'),(tw,th))
        if len(v_boxes)!=1 or len(t_boxes)!=1:n_skip+=1;continue
        nb_v=normalized_box(v_boxes[0],(vh,vw))
        nb_t=normalized_box(t_boxes[0],(th,tw))
        rows['before'].append([center_displacement(nb_v,nb_t),scale_ratio(nb_v,nb_t),alignment_iou(nb_v,nb_t)])
        outs=model(v.to(args.device),ir.to(args.device))
        grids=outs['alignment_aux']['sampling_grids']
        for i,g in enumerate(grids):
            box=map_thermal_box_to_visible(q['boxes'][0],g,t['image_size'],
                float(q['visible_ratio']),q['visible_pad'].tolist(),(vh,vw)).cpu()
            rows[f'full_p{i+2}'].append([center_displacement(box,nb_v),scale_ratio(box,nb_v),alignment_iou(box,nb_v)])
    print(json.dumps({'samples_skipped_due_to_missing_or_ambiguous_pairs':n_skip,
        'definitions':'centers and areas normalized in native modality dimensions; 5 sample mapped box approximation',
        'results':{key:summarize(values) for key,values in rows.items()}},indent=2))
if __name__=='__main__':main()
