"""Generate synthetic, non-project toy input. NEVER use to claim AeroMis-VT metrics."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def create(root, count=2, size=64):
    root=Path(root)
    for split in ('train','val','test'):
        for sub in ('visible','thermal','labels','visible_labels'):
            (root/split/sub).mkdir(parents=True,exist_ok=True)
        for idx in range(count):
            rng=np.random.default_rng(1000+idx+{'train':0,'val':50,'test':100}[split])
            image=Image.new('RGB',(size,size),(177,210,222))
            visible=ImageDraw.Draw(image)
            thermal_im=Image.new('RGB',(size,size),(18,20,30))
            thermal=ImageDraw.Draw(thermal_im)
            cx=size//2+int(rng.integers(-7,8));cy=size//2+int(rng.integers(-7,8));r=5
            visible.ellipse((cx-3-r,cy-1-r,cx-3+r,cy-1+r), fill=(50,68,74))
            thermal.ellipse((cx-r,cy-r,cx+r,cy+r), fill=(238,239,215))
            name=f'{idx:05d}'
            image.save(root/split/'visible'/f'{name}.png')
            thermal_im.save(root/split/'thermal'/f'{name}.png')
            (root/split/'labels'/f'{name}.txt').write_text(f'0 {cx/size:.8f} {cy/size:.8f} {2*r/size:.8f} {2*r/size:.8f}\n')
            (root/split/'visible_labels'/f'{name}.txt').write_text(f'0 {(cx-3)/size:.8f} {(cy-1)/size:.8f} {2*r/size:.8f} {2*r/size:.8f}\n')
    print(f'Synthetic demo created in {root.resolve()} (NOT AeroMis-VT).')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/demo'); p.add_argument('--count',type=int,default=2); p.add_argument('--size',type=int,default=64); a=p.parse_args();create(a.root,a.count,a.size)
