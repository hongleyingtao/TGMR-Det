"""Metrics for co-visible, explicitly one-to-one matched annotation pairs."""
import math
import torch
import torch.nn.functional as F


def normalized_box(box, size_hw):
    h, w = size_hw
    v = torch.as_tensor(box, dtype=torch.float32).clone()
    v[[0,2]] /= w
    v[[1,3]] /= h
    return v


def center_displacement(a, b):
    ca = (a[:2] + a[2:]) * .5
    cb = (b[:2] + b[2:]) * .5
    return float((ca-cb).norm())


def scale_ratio(a,b):
    def area(v):
        return float((v[2]-v[0]).clamp(min=0)*(v[3]-v[1]).clamp(min=0))
    aa, ab = area(a), area(b)
    if min(aa,ab) <= 0: raise ValueError('zero-area bbox')
    return max(aa/ab,ab/aa)


def alignment_iou(a,b):
    lt=torch.maximum(a[:2],b[:2]); rb=torch.minimum(a[2:],b[2:]); wh=(rb-lt).clamp(min=0)
    inter=float(wh[0]*wh[1]); aa=float((a[2]-a[0])*(a[3]-a[1])); ab=float((b[2]-b[0])*(b[3]-b[1]))
    return inter/(aa+ab-inter+1e-9)


def sample_grid_at_points(grid, points_xy, input_image_size):
    """Sample thermal->visible normalized grid at thermal-domain pixel points."""
    q=points_xy.clone().to(grid.device)
    q[:,0]=2.0*q[:,0]/input_image_size-1
    q[:,1]=2.0*q[:,1]/input_image_size-1
    s=F.grid_sample(grid.permute(0,3,1,2),q[None,:,None,:], mode='bilinear', padding_mode='border',align_corners=False)
    return s[0,:,:,0].T


def map_thermal_box_to_visible(thermal_box_input, grid, image_size, visible_ratio, visible_pad, visible_original_hw):
    """Push 5 samples (corners and center) through thermal->visible sampling grid.

    Under a nonlinear grid this is an APPROXIMATION to a warped bounding box;
    document this explicitly when reporting registration geometry results.
    """
    b=thermal_box_input.to(grid.device)
    x1,y1,x2,y2=b
    pts=torch.stack([torch.stack([(x1+x2)/2,(y1+y2)/2]),torch.stack([x1,y1]),torch.stack([x2,y1]),torch.stack([x2,y2]),torch.stack([x1,y2])])
    s=sample_grid_at_points(grid,pts,image_size)
    xy=(s+1)*.5*image_size
    xy[:,0]=(xy[:,0]-float(visible_pad[0]))/float(visible_ratio)
    xy[:,1]=(xy[:,1]-float(visible_pad[1]))/float(visible_ratio)
    h,w=visible_original_hw
    xy[:,0]/=w; xy[:,1]/=h
    return torch.stack([xy[:,0].min(),xy[:,1].min(),xy[:,0].max(),xy[:,1].max()])
