"""Transparent mAP@0.5 / mAP@0.5:0.95 and FPPI evaluator.

This is not a verification of the paper's historical experiment logs. Keep
score and NMS definitions fixed when comparing to other implementations.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tgmr_det.config import load_config
from tgmr_det.data import PairedVisibleThermalDataset, paired_collate_fn
from tgmr_det.factory import build_model
from tgmr_det.utils import load_checkpoint


def ap_from_pr(rec, prec):
    """101-point interpolated AP, explicit endpoint conventions."""
    if not len(rec): return 0.0
    mrec=np.r_[0.0,rec,1.0]
    mpre=np.r_[1.0,prec,0.0]
    mpre=np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.trapezoid(np.interp(np.linspace(0,1,101),mrec,mpre),np.linspace(0,1,101)))


def evaluate_predictions(predictions, targets, num_classes=1, fppi_threshold=.20):
    thresholds=np.round(np.arange(.50,.96,.05),2)
    aps_by_iou=[]
    total_pred_over_cut=0
    for pred in predictions:
        total_pred_over_cut+=int((pred['scores']>=fppi_threshold).sum())
    # Count false positives per image at fixed score threshold independently of AP.
    false_positives=0
    tp_at_cut=0
    gt_count=0
    for pred, gt in zip(predictions,targets):
        gt_count+=len(gt['boxes'])
        used=torch.zeros(len(gt['boxes']),dtype=torch.bool)
        keep=torch.where(pred['scores']>=fppi_threshold)[0]
        for j in keep.tolist():
            if len(gt['boxes'])==0: false_positives+=1;continue
            iou=box_iou(pred['boxes'][j:j+1],gt['boxes']).flatten()
            if len(iou):
                iou[used]=-1
                k=int(iou.argmax())
                if iou[k]>=.5:used[k]=True;tp_at_cut+=1;continue
            false_positives+=1
    for threshold in thresholds:
        perclass=[]
        for cls in range(num_classes):
            items=[];n_gt=0;gt_by_img=[]
            for idx,(pred,gt) in enumerate(zip(predictions,targets)):
                g=gt['boxes'][gt['labels']==cls]; n_gt+=len(g)
                gt_by_img.append(g)
                for b,s in zip(pred['boxes'][pred['labels']==cls],pred['scores'][pred['labels']==cls]):
                    items.append((float(s),idx,b))
            if n_gt==0:continue
            items.sort(key=lambda row: row[0],reverse=True)
            used=[torch.zeros(len(b),dtype=torch.bool) for b in gt_by_img]
            tp=np.zeros(len(items),dtype=np.float64)
            fp=np.zeros(len(items),dtype=np.float64)
            for i,(_,img,box) in enumerate(items):
                g=gt_by_img[img]
                if not len(g):fp[i]=1;continue
                ious=box_iou(box[None],g).flatten()
                ious[used[img]]=-1.0
                best=int(ious.argmax())
                if float(ious[best])>=threshold:
                    tp[i]=1;used[img][best]=True
                else:fp[i]=1
            if len(items):
                precision=np.cumsum(tp)/np.maximum(np.cumsum(tp)+np.cumsum(fp),1e-12)
                recall=np.cumsum(tp)/n_gt
                perclass.append(ap_from_pr(recall,precision))
            else:perclass.append(0.0)
        aps_by_iou.append(float(np.mean(perclass)) if perclass else None)
    good=[v for v in aps_by_iou if v is not None]
    return {
        'images':len(predictions),'thermal_gt':gt_count,
        'map50':aps_by_iou[0],
        'map50_95':float(np.mean(good)) if good else None,
        'recall_at_score_020':tp_at_cut/max(gt_count,1) if gt_count else None,
        'fppi_at_score_020':false_positives/max(1,len(predictions)),
        'precision_at_score_020':tp_at_cut/max(1,tp_at_cut+false_positives),
        'iou_thresholds':[float(x) for x in thresholds],
        'ap_by_iou':aps_by_iou,
        'eval_note':'AP computed from predictions with config eval confidence threshold; recall/FPPI at score 0.20.'
    }


@torch.no_grad()
def collect(cfg,weights,split,device):
    d,t=cfg['data'],cfg['train']
    cfg['model']['pretrained_yolo11']=None
    model=build_model(cfg).to(device)
    load_checkpoint(weights,model,map_location=device)
    model.eval()
    dataset=PairedVisibleThermalDataset(
        d['root'],split,t['image_size'],d['visible_dir'],d['thermal_dir'],
        d['labels_dir'],d['image_extensions'],training=False)
    loader=DataLoader(dataset,batch_size=t['batch_size'],shuffle=False,
        num_workers=cfg.get('num_workers',0),collate_fn=paired_collate_fn)
    preds,targets=[],[]
    for v,ir,gt in loader:
        output=model(v.to(device),ir.to(device))
        dets=model.decode(output, conf_threshold=cfg['eval']['conf_threshold'],
            nms_iou_threshold=cfg['eval']['nms_iou_threshold'],
            max_detections=cfg['eval']['max_detections'])
        for x,y in zip(dets,gt):
            preds.append({k:x[k].detach().cpu() for k in ('boxes','scores','labels')})
            targets.append({'boxes':y['boxes'].cpu(),'labels':y['labels'].cpu()})
    return evaluate_predictions(preds,targets,model.num_classes)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/aeromis_vt.yaml')
    p.add_argument('--weights',required=True)
    p.add_argument('--split',default='test')
    p.add_argument('--data-root',default=None)
    p.add_argument('--device',default='cpu')
    args=p.parse_args(); cfg=load_config(args.config)
    if args.data_root:cfg['data']['root']=args.data_root
    print(json.dumps(collect(cfg,args.weights,args.split,torch.device(args.device)),indent=2))
if __name__=='__main__':main()
