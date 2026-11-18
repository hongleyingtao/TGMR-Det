"""Deterministic EO-only CGM / CGLM generator according to the manuscript.

Input layout: root/{train,val,test}/{visible,thermal,labels} with matching names.
All thermal frames and thermal ground truth remain UNALTERED. Synthetic
perturbations are suitable only for the public EO--IR benchmark when permitted.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import cv2
import numpy as np

SEVERITIES={
 'light':(.01,1.0,.98,1.02,.0025),
 'medium':(.03,3.0,.95,1.05,.0050),
 'heavy':(.05,5.0,.90,1.10,.0100),
}


def perturb_visible(image,rng,protocol='cgm',severity='light'):
    if protocol not in ('cgm','cglm'):raise ValueError(protocol)
    shift,rotate,smin,smax,local_ratio=SEVERITIES[severity]
    h,w=image.shape[:2];center=np.array([(w-1)*.5,(h-1)*.5],dtype=np.float64)
    sx,sy=rng.uniform(smin,smax,2)
    rot=float(rng.uniform(-rotate,rotate))
    dx=float(rng.uniform(-shift*w,shift*w));dy=float(rng.uniform(-shift*h,shift*h))
    theta=np.deg2rad(rot)
    R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    A=np.diag([sx,sy])@R
    t=center-A@center+np.array([dx,dy])
    M=np.column_stack([A,t]).astype(np.float32)
    affine=cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101)
    note={'tx_frac':dx/w,'ty_frac':dy/h,'angle_deg':rot,'sx':float(sx),'sy':float(sy),
          'max_local_fraction_short_side':local_ratio if protocol=='cglm' else 0.0}
    if protocol=='cgm':return affine,note
    amplitude=local_ratio*min(w,h)
    dx_grid=rng.uniform(-amplitude,amplitude,(10,10)).astype(np.float32)
    dy_grid=rng.uniform(-amplitude,amplitude,(10,10)).astype(np.float32)
    sigma=max(w,h)/80.0
    local_x=cv2.GaussianBlur(cv2.resize(dx_grid,(w,h),interpolation=cv2.INTER_CUBIC),(0,0),sigmaX=sigma,sigmaY=sigma)
    local_y=cv2.GaussianBlur(cv2.resize(dy_grid,(w,h),interpolation=cv2.INTER_CUBIC),(0,0),sigmaX=sigma,sigmaY=sigma)
    yy,xx=np.indices((h,w),dtype=np.float32)
    result=cv2.remap(affine,xx+local_x,yy+local_y,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return result,note


def build(source,dest,protocol,severity,seed):
    source,dest=Path(source).resolve(),Path(dest).resolve()
    if source==dest or source in dest.parents:raise ValueError('Output must be outside the input dataset tree')
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Output directory must be empty')
    rng=np.random.default_rng(seed)
    samples=[]
    for split in ('train','val','test'):
        src=source/split
        if not src.is_dir():continue
        for folder in ('visible','thermal','labels'):
            if not (src/folder).is_dir():raise FileNotFoundError(src/folder)
            (dest/split/folder).mkdir(parents=True,exist_ok=True)
        for img_path in sorted((src/'visible').iterdir()):
            if img_path.suffix.lower() not in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):continue
            image=cv2.imread(str(img_path),cv2.IMREAD_UNCHANGED)
            if image is None:raise RuntimeError(f'Unreadable file: {img_path}')
            result,parameters=perturb_visible(image,rng,protocol,severity)
            if not cv2.imwrite(str(dest/split/'visible'/img_path.name),result):raise IOError(img_path)
            # Resolve by shared stem; do not invent missing pair/label files.
            thermal=list((src/'thermal').glob(img_path.stem+'.*'))
            if len(thermal)!=1:raise RuntimeError(f'{img_path.stem}: expected one thermal match, got {len(thermal)}')
            shutil.copy2(thermal[0],dest/split/'thermal'/thermal[0].name)
            label=src/'labels'/(img_path.stem+'.txt')
            if not label.exists():raise FileNotFoundError(label)
            shutil.copy2(label,dest/split/'labels'/label.name)
            samples.append({'split':split,'visible':img_path.name,'thermal':thermal[0].name,**parameters})
    # Only geometry parameters and filenames, not project images.
    (dest/'protocol_metadata.json').write_text(json.dumps({
        'protocol':protocol,'severity':severity,'seed':seed,'pairs':len(samples),
        'sampling':'numpy.default_rng with sorted filenames',
        'warping':'center-based affine; bilinear INTER_LINEAR; BORDER_REFLECT_101',
        'local':'10x10 uniform control grid -> bicubic -> Gaussian sigma=max(W,H)/80 -> remap',
    },indent=2))
    (dest/'perturbations.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    print(f'Generated {len(samples)} EO-only perturbed pairs under {dest}')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--dest',required=True)
    p.add_argument('--protocol',choices=('cgm','cglm'),required=True)
    p.add_argument('--severity',choices=tuple(SEVERITIES),required=True)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args();build(a.source,a.dest,a.protocol,a.severity,a.seed)
