import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
import pytest
from tgmr_det.config import load_config
from tgmr_det.factory import build_model
from tgmr_det.model import MSFRNet, ITSM, SSFCFM
from tgmr_det.losses import DetectionLoss
from tgmr_det.geometry import center_displacement,scale_ratio


def test_registration_exact_identity_and_grid():
    torch.set_num_threads(1);torch.manual_seed(5)
    module=MSFRNet(8).eval()
    v=torch.randn(1,8,16,20);t=torch.randn_like(v)
    with torch.no_grad():warped,affine,flow,grid=module(v,t)
    assert torch.allclose(warped,v,atol=2e-5)
    assert torch.count_nonzero(flow)==0
    assert torch.allclose(affine, torch.tensor([[0.,0.,1.,1.]]))
    assert grid.shape==(1,16,20,2)


def test_model_and_loss_backward():
    torch.set_num_threads(1);torch.manual_seed(4)
    cfg=load_config(ROOT/'configs/demo.yaml')
    m=build_model(cfg).train()
    v=torch.rand(1,3,64,64);t=torch.rand(1,3,64,64)
    out=m(v,t)
    assert [x['cls'].shape[-1] for x in out['levels']]==[16,8,4,2]
    assert len(out['alignment_aux']['sampling_grids'])==4
    c=DetectionLoss(1,m.strides,cfg['model']['regression_ranges'],
        cls_weight=1.,ctr_weight=1.,box_weight=2.,align_weight=.02,
        flow_smooth_weight=1.,flow_magnitude_weight=.05,affine_weight=.02)
    result=c(m,out,[{'boxes':torch.tensor([[26.,26.,38.,38.]]),'labels':torch.tensor([0])}])
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert m.fusion.msfr[0].regressor.weight.grad is not None
    with torch.no_grad():assert len(m.decode(out,conf_threshold=.99))==1


def test_ablation_head_flags():
    torch.set_num_threads(1)
    cfg=load_config(ROOT/'configs/demo.yaml')
    cfg['model']['use_p2_head']=False
    cfg['model']['head_variant']='shared'
    m=build_model(cfg).eval()
    with torch.no_grad():out=m(torch.randn(1,3,64,64),torch.randn(1,3,64,64))
    assert len(out['levels'])==3
    assert m.strides==(8,16,32)


def test_bbox_geometry():
    a=torch.tensor([.1,.1,.3,.3]);b=torch.tensor([.1,.1,.3,.3]);c=torch.tensor([.2,.1,.4,.3])
    assert center_displacement(a,b)<1e-6
    assert abs(scale_ratio(a,b)-1.)<1e-6
    assert center_displacement(a,c)>.09


def test_module_ablation_options():
    cfg=load_config(ROOT/'configs/demo.yaml')
    cfg['model'].update(use_itsm=False, use_msfr=False, use_ssfcfm=False)
    m=build_model(cfg).eval()
    with torch.no_grad():out=m(torch.rand(1,3,64,64),torch.rand(1,3,64,64))
    assert len(out['levels'])==4
    assert out['alignment_aux']['flows'][0].abs().sum().item()==0
