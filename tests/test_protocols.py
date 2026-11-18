import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from tools.generate_cgm_cglm import perturb_visible, SEVERITIES

def test_cgm_reproducible_and_nontrivial():
    im=np.zeros((64,96,3),dtype=np.uint8)
    im[20:25,40:50]=255
    a,n1=perturb_visible(im,np.random.default_rng(2026),'cgm','heavy')
    b,n2=perturb_visible(im,np.random.default_rng(2026),'cgm','heavy')
    assert np.array_equal(a,b)
    assert n1==n2
    assert np.sum(np.abs(a.astype(float)-im.astype(float)))>0

def test_cglm_reproducible():
    im=np.zeros((96,96,3),dtype=np.uint8)
    im[40:60,35:65]=255
    a,_=perturb_visible(im,np.random.default_rng(2026),'cglm','medium')
    b,_=perturb_visible(im,np.random.default_rng(2026),'cglm','medium')
    assert np.array_equal(a,b)
