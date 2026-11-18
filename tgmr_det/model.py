from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple
import copy

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from .s3a import StructuredSparseSelfAttentionBlock
from .backbones import OfficialYOLO11nBackbone


def make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvGNAct(nn.Module):
    """Batch-size-1-safe convolution used by the affine regressor."""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        groups = min(8, c2)
        while c2 % groups != 0:
            groups -= 1
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.norm = nn.GroupNorm(groups, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        hidden = c // 2
        self.cv1 = ConvBNAct(c, hidden, 1)
        self.cv2 = ConvBNAct(hidden, c, 3)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class CSPBlock(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        hidden = c2 // 2
        self.cv1 = ConvBNAct(c1, hidden, 1)
        self.cv2 = ConvBNAct(c1, hidden, 1)
        self.blocks = nn.Sequential(*[Bottleneck(hidden) for _ in range(n)])
        self.cv3 = ConvBNAct(hidden * 2, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), dim=1))


class C3K(nn.Module):
    """YOLO11-style compact C3k block with two 3x3 convolutions per bottleneck."""

    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = ConvBNAct(c, c, 3)
        self.cv2 = ConvBNAct(c, c, 3)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C3k2(nn.Module):
    """
    Lightweight YOLO11 C3k2-style block.

    A lightweight C3k2-style aggregator in the custom PAN-FPN neck;
    it is not the official Ultralytics C3k2 implementation.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, expansion=0.5):
        super().__init__()
        hidden = make_divisible(c2 * expansion)
        self.cv1 = ConvBNAct(c1, hidden * 2, 1)
        self.blocks = nn.ModuleList([C3K(hidden, shortcut=shortcut) for _ in range(n)])
        self.cv2 = ConvBNAct(hidden * (2 + n), c2, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class CSPDemoBackbone(nn.Module):
    """DEMO ONLY: CSP-style surrogate, not the YOLO11 backbone claimed in the paper."""

    def __init__(self, width=0.75):
        super().__init__()
        c1, c2, c3, c4, c5 = [make_divisible(x * width) for x in (32, 64, 128, 256, 512)]
        self.channels = (c2, c3, c4, c5)
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.s2 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), CSPBlock(c2, c2, 1))
        self.s3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), CSPBlock(c3, c3, 2))
        self.s4 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), CSPBlock(c4, c4, 2))
        self.s5 = nn.Sequential(ConvBNAct(c4, c5, 3, 2), CSPBlock(c5, c5, 1))

    def forward(self, x):
        x = self.stem(x)
        p2 = self.s2(x)
        p3 = self.s3(p2)
        p4 = self.s4(p3)
        p5 = self.s5(p4)
        return [p2, p3, p4, p5]


class ITSM(nn.Module):
    """
    Thermal queries globally search visible features on a pooled token grid.
    Cross-attention output is arranged in thermal-query positions.
    """

    def __init__(self, channels, dim=96, heads=4, pool_size=16):
        super().__init__()
        dim = min(dim, channels)
        dim = max(heads, dim // heads * heads)
        self.pool_size = int(pool_size)
        # The paper's learnable positional encodings E_v and E_t.
        self.visible_pos = nn.Parameter(torch.zeros(1, dim, self.pool_size, self.pool_size))
        self.thermal_pos = nn.Parameter(torch.zeros(1, dim, self.pool_size, self.pool_size))
        nn.init.normal_(self.visible_pos, std=0.02)
        nn.init.normal_(self.thermal_pos, std=0.02)
        self.v_reduce = ConvBNAct(channels, dim, 1)
        self.t_reduce = ConvBNAct(channels, dim, 1)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.v_out = ConvBNAct(dim, channels, 1)
        self.t_out = ConvBNAct(dim, channels, 1)
        self.v_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.t_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())

    def _tokens(self, x, pos):
        h = min(self.pool_size, x.shape[-2])
        w = min(self.pool_size, x.shape[-1])
        pooled = F.adaptive_avg_pool2d(x, (h, w))
        pos_map = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        return (pooled + pos_map).flatten(2).transpose(1, 2), (h, w)

    def forward(self, visible, thermal):
        vr = self.v_reduce(visible)
        tr = self.t_reduce(thermal)
        vt, hw = self._tokens(vr, self.visible_pos)
        tt, thermal_hw = self._tokens(tr, self.thermal_pos)
        if hw != thermal_hw:
            raise ValueError("Cross-attention must use the same pooled reference grid")

        cross, _ = self.cross_attn(tt, vt, vt, need_weights=False)
        thermal_self, _ = self.self_attn(tt, tt, tt, need_weights=False)

        b, _, d = cross.shape
        h, w = hw
        cross_map = cross.transpose(1, 2).reshape(b, d, h, w)
        self_map = thermal_self.transpose(1, 2).reshape(b, d, h, w)
        cross_map = F.interpolate(cross_map, visible.shape[-2:], mode="bilinear", align_corners=False)
        self_map = F.interpolate(self_map, thermal.shape[-2:], mode="bilinear", align_corners=False)

        v_context = self.v_out(cross_map)
        t_context = self.t_out(self_map)
        visible = visible + self.v_gate(torch.cat([visible, v_context], 1)) * v_context
        thermal = thermal + self.t_gate(torch.cat([thermal, t_context], 1)) * t_context
        return visible, thermal


class MSFRNet(nn.Module):
    """
    MSFR-Net: scale-specific global translation/scale plus bounded residual flow.
    Zero initialization guarantees identity sampling at startup.
    """

    def __init__(self, channels, max_shift=0.25, max_log_scale=0.20, max_local_shift=2.0):
        super().__init__()
        hidden = max(32, channels // 2)
        self.diff_projection = ConvGNAct(channels, channels, 1, 1)
        self.matcher = nn.Sequential(
            ConvGNAct(channels, hidden, 3, 2),
            ConvGNAct(hidden, hidden, 3, 2),
            ConvGNAct(hidden, hidden, 3, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Linear(hidden, 4)
        nn.init.zeros_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)
        self.local_flow = nn.Sequential(
            ConvGNAct(channels * 3, hidden, 3, 1),
            ConvGNAct(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )
        nn.init.zeros_(self.local_flow[-1].weight)
        nn.init.zeros_(self.local_flow[-1].bias)
        self.max_shift = float(max_shift)
        self.max_log_scale = float(max_log_scale)
        self.max_local_shift = float(max_local_shift)

    def forward(self, visible, thermal):
        diff = self.diff_projection(visible - thermal)
        z = self.matcher(diff).flatten(1)
        raw = self.regressor(z)
        tx_ty = torch.tanh(raw[:, :2]) * self.max_shift
        scales = torch.exp(torch.tanh(raw[:, 2:]) * self.max_log_scale)

        theta = visible.new_zeros((visible.shape[0], 2, 3))
        theta[:, 0, 0] = scales[:, 0]
        theta[:, 1, 1] = scales[:, 1]
        theta[:, 0, 2] = tx_ty[:, 0]
        theta[:, 1, 2] = tx_ty[:, 1]
        grid = F.affine_grid(theta, thermal.shape, align_corners=False)
        global_warp = F.grid_sample(
            visible, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

        flow_px = torch.tanh(
            self.local_flow(torch.cat([global_warp, thermal, diff], dim=1))
        ) * self.max_local_shift
        # grid_sample/affine_grid both use align_corners=False. For a feature
        # pixel displacement, one pixel corresponds to 2/W or 2/H, not 2/(W-1).
        b, _, h, w = flow_px.shape
        flow_norm = torch.stack(
            [flow_px[:, 0] * (2.0 / w), flow_px[:, 1] * (2.0 / h)], dim=-1
        )
        # The paper defines ONE resampling operation: Warp(F_v, Gamma_g + R).
        # Avoid an unintended second interpolation through global_warp.
        full_grid = grid + flow_norm
        warped = F.grid_sample(
            visible, full_grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        summary = torch.cat([tx_ty, scales], dim=1)
        return warped, summary, flow_px, full_grid


class SSFCFM(nn.Module):
    """Bi-directional channel/spatial correction and fusion."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1),
        )
        self.spatial = nn.Conv2d(4, 2, 7, padding=3)
        self.out = CSPBlock(channels, channels, 1)

    def forward(self, visible, thermal):
        cat = torch.cat([visible, thermal], dim=1)
        pooled = torch.cat([F.adaptive_avg_pool2d(cat, 1), F.adaptive_max_pool2d(cat, 1)], dim=1)
        cv, ct = torch.sigmoid(self.channel_mlp(pooled)).chunk(2, dim=1)
        v1 = visible + cv * visible
        t1 = thermal + ct * thermal
        stats = torch.cat(
            [v1.mean(1, keepdim=True), v1.amax(1, keepdim=True),
             t1.mean(1, keepdim=True), t1.amax(1, keepdim=True)],
            dim=1,
        )
        sv, st = torch.sigmoid(self.spatial(stats)).chunk(2, dim=1)
        return self.out((v1 + sv * v1) + (t1 + st * t1))



class IdentityITSM(nn.Module):
    """Ablation-only: bypass thermal-guided search."""
    def forward(self, visible, thermal):
        return visible, thermal


class IdentityMSFR(nn.Module):
    """Ablation-only: no warp; retain complete output signature."""
    def forward(self, visible, thermal):
        b, _, h, w = thermal.shape
        theta = thermal.new_zeros((b, 2, 3))
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        grid = F.affine_grid(theta, thermal.shape, align_corners=False)
        affine = torch.stack([theta[:, 0, 2], theta[:, 1, 2],
                              theta[:, 0, 0], theta[:, 1, 1]], dim=1)
        flow = thermal.new_zeros((b, 2, h, w))
        return visible, affine, flow, grid


class AdditiveFusion(nn.Module):
    """Ablation-only: direct equal-weight fusion when SSFCFM is disabled."""
    def forward(self, visible, thermal):
        return 0.5 * (visible + thermal)


class TGMRFusionPyramid(nn.Module):
    def __init__(self, channels, itsm_dim, itsm_heads, itsm_pool, max_shift, max_log_scale, max_local_shift=2.0, use_itsm=True, use_msfr=True, use_ssfcfm=True):
        super().__init__()
        self.itsm = nn.ModuleList(
            [ITSM(c, itsm_dim, itsm_heads, itsm_pool) if use_itsm else IdentityITSM() for c in channels]
        )
        self.msfr = nn.ModuleList(
            [MSFRNet(c, max_shift, max_log_scale, max_local_shift) if use_msfr else IdentityMSFR() for c in channels]
        )
        self.ssfcfm = nn.ModuleList([SSFCFM(c) if use_ssfcfm else AdditiveFusion() for c in channels])

    def forward(self, visible_features, thermal_features):
        fused, params, flows, grids = [], [], [], []
        for v, t, itsm, msfr, ssfcfm in zip(
            visible_features, thermal_features, self.itsm, self.msfr, self.ssfcfm
        ):
            v, t = itsm(v, t)
            v, affine, flow, full_grid = msfr(v, t)
            fused.append(ssfcfm(v, t))
            params.append(affine)
            flows.append(flow)
            grids.append(full_grid)
        return fused, params, flows, grids


class TGMRPANFPN(nn.Module):
    """
    PAN-FPN using ITSM -> MSFR-Net -> SSFCFM fused P2--P5 features.
    Structured sparse self-attention is applied to P2--P4 only.
    """

    def __init__(self, in_channels, out_channels=128, sparse_attention=True, sparse_group_size=2, heads=4):
        super().__init__()
        self.sparse_attention = bool(sparse_attention)
        self.lateral = nn.ModuleList([ConvBNAct(c, out_channels, 1) for c in in_channels])
        self.top_blocks = nn.ModuleList(
            [C3k2(out_channels * 2, out_channels, n=1, shortcut=False) for _ in range(3)]
        )
        self.down = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, 3, 2) for _ in range(3)]
        )
        self.bottom_blocks = nn.ModuleList(
            [C3k2(out_channels * 2, out_channels, n=1, shortcut=False) for _ in range(3)]
        )
        if sparse_attention:
            self.psa_p2 = StructuredSparseSelfAttentionBlock(out_channels, out_channels, n=1, sparse_size=sparse_group_size)
            self.psa_p3 = StructuredSparseSelfAttentionBlock(out_channels, out_channels, n=1, sparse_size=sparse_group_size)
            self.psa_p4 = StructuredSparseSelfAttentionBlock(out_channels, out_channels, n=1, sparse_size=sparse_group_size)
        else:
            self.psa_p2 = nn.Identity()
            self.psa_p3 = nn.Identity()
            self.psa_p4 = nn.Identity()

    def forward(self, feats):
        p2, p3, p4, p5 = [layer(x) for layer, x in zip(self.lateral, feats)]
        f4 = self.top_blocks[0](
            torch.cat([p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")], 1)
        )
        f3 = self.top_blocks[1](
            torch.cat([p3, F.interpolate(f4, size=p3.shape[-2:], mode="nearest")], 1)
        )
        f2 = self.top_blocks[2](
            torch.cat([p2, F.interpolate(f3, size=p2.shape[-2:], mode="nearest")], 1)
        )
        f2 = self.psa_p2(f2)
        o3 = self.bottom_blocks[0](torch.cat([f3, self.down[0](f2)], 1))
        o3 = self.psa_p3(o3)
        o4 = self.bottom_blocks[1](torch.cat([f4, self.down[1](o3)], 1))
        o4 = self.psa_p4(o4)
        o5 = self.bottom_blocks[2](torch.cat([p5, self.down[2](o4)], 1))
        return [f2, o3, o4, o5]

class TGMRDetectionHead(nn.Module):
    """
    YOLO11-style anchor-free detection head.

    ``decoupled`` is the full TGMR-Det head, with separate classification and
    regression/centerness towers. ``shared`` is used only as a controlled
    ablation: it shares one feature tower while preserving prediction outputs,
    loss functions, feature levels, and decoding.
    """

    def __init__(self, channels, num_classes, head_variant="decoupled"):
        super().__init__()
        self.head_variant = str(head_variant).lower()

        if self.head_variant == "decoupled":
            self.cls_tower = nn.Sequential(
                C3k2(channels, channels, n=1, shortcut=False),
                ConvBNAct(channels, channels, 3),
            )
            self.reg_tower = nn.Sequential(
                C3k2(channels, channels, n=1, shortcut=False),
                ConvBNAct(channels, channels, 3),
            )
        elif self.head_variant == "shared":
            self.shared_tower = nn.Sequential(
                C3k2(channels, channels, n=1, shortcut=False),
                ConvBNAct(channels, channels, 3),
            )
        else:
            raise ValueError(
                f"Unsupported head_variant={head_variant!r}; "
                "expected 'decoupled' or 'shared'."
            )

        self.cls_pred = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.ctr_pred = nn.Conv2d(channels, 1, 3, padding=1)
        self.reg_pred = nn.Conv2d(channels, 4, 3, padding=1)

        prior = 0.01
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - prior) / prior))
        nn.init.constant_(self.ctr_pred.bias, -math.log((1 - prior) / prior))
        nn.init.constant_(self.reg_pred.bias, 1.0)

    def forward(self, x):
        if self.head_variant == "decoupled":
            c = self.cls_tower(x)
            r = self.reg_tower(x)
            return {
                "cls": self.cls_pred(c),
                "ctr": self.ctr_pred(r),
                "reg": self.reg_pred(r),
            }

        f = self.shared_tower(x)
        return {
            "cls": self.cls_pred(f),
            "ctr": self.ctr_pred(f),
            "reg": self.reg_pred(f),
        }


class TGMRDet(nn.Module):
    """
    TGMR-Det: two independent backbones, ITSM, MSFR-Net, SSFCFM, S3A, FCOS head.
    Predictions are always in the thermal reference coordinate system.
    """

    def __init__(
        self,
        num_classes=1,
        width=0.75,
        neck_channels=128,
        itsm_dim=96,
        itsm_heads=4,
        itsm_pool=16,
        max_shift=0.25,
        max_log_scale=0.20,
        max_local_shift=2.0,
        sparse_attention=True,
        sparse_group_size=2,
        use_itsm=True,
        use_msfr=True,
        use_ssfcfm=True,
        backbone_type="official_yolo11n",
        pretrained_yolo11=None,
        strides=(4, 8, 16, 32),
        use_p2_head=True,
        head_variant="decoupled",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_p2_head = bool(use_p2_head)

        all_strides = tuple(int(x) for x in strides)
        if len(all_strides) != 4:
            raise ValueError(
                f"TGMRDet expects P2--P5 strides, but received {all_strides}."
            )

        # Full model: P2--P5; ablated model: P3--P5 only.
        # P2 remains inside the neck for feature fusion.
        self.strides = all_strides if self.use_p2_head else all_strides[1:]
        if backbone_type == "official_yolo11n":
            source_backbone = OfficialYOLO11nBackbone(pretrained_yolo11)
        elif backbone_type == "csp_demo":
            if pretrained_yolo11:
                raise ValueError("csp_demo cannot use official pretrained YOLO11 weights")
            source_backbone = CSPDemoBackbone(width)
        else:
            raise ValueError("backbone_type must be 'official_yolo11n' or 'csp_demo'")
        self.backbone_type = backbone_type
        self.visible_backbone = source_backbone
        self.thermal_backbone = copy.deepcopy(source_backbone)
        channels = self.visible_backbone.channels
        self.fusion = TGMRFusionPyramid(
            channels, itsm_dim, itsm_heads, itsm_pool, max_shift, max_log_scale,
            max_local_shift, use_itsm=use_itsm, use_msfr=use_msfr, use_ssfcfm=use_ssfcfm
        )
        self.neck = TGMRPANFPN(
            channels,
            neck_channels,
            sparse_attention=sparse_attention,
            sparse_group_size=sparse_group_size,
            heads=itsm_heads,
        )
        self.head_variant = str(head_variant).lower()
        self.heads = nn.ModuleList(
            [
                TGMRDetectionHead(
                    neck_channels,
                    self.num_classes,
                    head_variant=self.head_variant,
                )
                for _ in self.strides
            ]
        )

    def forward(self, visible, thermal):
        vf = self.visible_backbone(visible)
        tf = self.thermal_backbone(thermal)
        fused, affine_params, flow_fields, grids = self.fusion(vf, tf)
        pyramid = self.neck(fused)

        # The neck always returns P2--P5. For the no-P2 ablation,
        # retain P2 for neck fusion but remove its prediction output.
        if not self.use_p2_head:
            pyramid = pyramid[1:]

        if len(pyramid) != len(self.heads):
            raise RuntimeError(
                f"Feature/head mismatch: {len(pyramid)} feature levels versus "
                f"{len(self.heads)} detection heads."
            )

        return {
            "levels": [head(x) for head, x in zip(self.heads, pyramid)],
            "affine_params": affine_params,
            "alignment_aux": {"flows": flow_fields, "sampling_grids": grids},
        }

    @staticmethod
    def make_points(h, w, stride, device, dtype):
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack(
            ((x + 0.5) * stride, (y + 0.5) * stride), dim=-1
        ).reshape(-1, 2)

    def decode(
        self,
        outputs,
        conf_threshold=0.25,
        nms_iou_threshold=0.5,
        max_detections=100,
    ):
        levels = outputs["levels"]
        batch = levels[0]["cls"].shape[0]
        all_boxes = [[] for _ in range(batch)]
        all_scores = [[] for _ in range(batch)]
        all_labels = [[] for _ in range(batch)]

        for stride, pred in zip(self.strides, levels):
            cls = pred["cls"].sigmoid()
            ctr = pred["ctr"].sigmoid()
            reg = F.softplus(pred["reg"]) * stride
            b, c, h, w = cls.shape
            points = self.make_points(h, w, stride, cls.device, cls.dtype)
            cls = cls.permute(0, 2, 3, 1).reshape(b, -1, c)
            ctr = ctr.permute(0, 2, 3, 1).reshape(b, -1, 1)
            reg = reg.permute(0, 2, 3, 1).reshape(b, -1, 4)
            scores = torch.sqrt(cls * ctr)

            for bi in range(batch):
                k = min(3000, scores[bi].numel())
                flat_scores, flat_idx = scores[bi].reshape(-1).topk(k, sorted=False)
                keep = flat_scores >= conf_threshold
                flat_scores, flat_idx = flat_scores[keep], flat_idx[keep]
                if flat_scores.numel() == 0:
                    continue
                point_idx = torch.div(flat_idx, c, rounding_mode="floor")
                labels = flat_idx % c
                p = points[point_idx]
                d = reg[bi, point_idx]
                boxes = torch.stack(
                    [p[:, 0] - d[:, 0], p[:, 1] - d[:, 1],
                     p[:, 0] + d[:, 2], p[:, 1] + d[:, 3]],
                    dim=1,
                )
                all_boxes[bi].append(boxes)
                all_scores[bi].append(flat_scores)
                all_labels[bi].append(labels)

        results = []
        for bi in range(batch):
            if not all_boxes[bi]:
                ref = levels[0]["cls"]
                results.append({
                    "boxes": ref.new_zeros((0, 4)),
                    "scores": ref.new_zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long, device=ref.device),
                })
                continue
            boxes = torch.cat(all_boxes[bi])
            scores = torch.cat(all_scores[bi])
            labels = torch.cat(all_labels[bi])
            keep = batched_nms(boxes, scores, labels, nms_iou_threshold)[:max_detections]
            results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            })
        return results
