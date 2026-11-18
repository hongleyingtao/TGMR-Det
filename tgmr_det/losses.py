from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def aligned_giou(boxes1, boxes2, eps=1e-7):
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1 + area2 - inter
    iou = inter / (union + eps)
    c_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    c_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c_area = c_wh[:, 0] * c_wh[:, 1]
    return iou - (c_area - union) / (c_area + eps)


def nwd_similarity(boxes1, boxes2, constant=12.8):
    c1 = (boxes1[:, :2] + boxes1[:, 2:]) * 0.5
    c2 = (boxes2[:, :2] + boxes2[:, 2:]) * 0.5
    s1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0)
    s2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0)
    distance = ((c1 - c2) ** 2).sum(1) + 0.25 * ((s1 - s2) ** 2).sum(1)
    return torch.exp(-torch.sqrt(distance.clamp(min=1e-12)) / constant)


def centerness_target(reg_targets, eps=1e-7):
    l, t, r, b = reg_targets.unbind(dim=1)
    lr = torch.minimum(l, r) / (torch.maximum(l, r) + eps)
    tb = torch.minimum(t, b) / (torch.maximum(t, b) + eps)
    return torch.sqrt((lr * tb).clamp(min=0))


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        strides,
        regression_ranges,
        center_sampling_radius=1.5,
        cls_weight=1.0,
        ctr_weight=1.0,
        box_weight=2.0,
        giou_weight=0.65,
        nwd_weight=0.35,
        nwd_constant=12.8,
        align_weight=0.02,
        flow_smooth_weight=1.0,
        flow_magnitude_weight=0.05,
        affine_weight=0.02,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = tuple(int(x) for x in strides)
        self.ranges = [tuple(float(v) for v in x) for x in regression_ranges]
        if len(self.ranges) != len(self.strides):
            raise ValueError("regression_ranges and strides must have equal length")
        self.center_sampling_radius = float(center_sampling_radius)
        self.cls_weight = float(cls_weight)
        self.ctr_weight = float(ctr_weight)
        self.box_weight = float(box_weight)
        self.giou_weight = float(giou_weight)
        self.nwd_weight = float(nwd_weight)
        self.nwd_constant = float(nwd_constant)
        self.align_weight = float(align_weight)
        self.flow_smooth_weight = float(flow_smooth_weight)
        self.flow_magnitude_weight = float(flow_magnitude_weight)
        self.affine_weight = float(affine_weight)

    def assign_level(self, points, boxes, labels, stride, reg_range):
        n = points.shape[0]
        device = points.device
        cls_targets = torch.zeros((n, self.num_classes), device=device)
        reg_targets = torch.zeros((n, 4), device=device)
        positive = torch.zeros((n,), dtype=torch.bool, device=device)
        if boxes.numel() == 0:
            return cls_targets, reg_targets, positive

        px, py = points[:, 0:1], points[:, 1:2]
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        l = px - x1[None, :]
        t = py - y1[None, :]
        r = x2[None, :] - px
        b = y2[None, :] - py
        ltrb = torch.stack([l, t, r, b], dim=2)
        inside_box = ltrb.amin(dim=2) > 0

        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        radius = self.center_sampling_radius * stride
        cx1 = torch.maximum(centers[:, 0] - radius, x1)
        cy1 = torch.maximum(centers[:, 1] - radius, y1)
        cx2 = torch.minimum(centers[:, 0] + radius, x2)
        cy2 = torch.minimum(centers[:, 1] + radius, y2)
        inside_center = (
            (px >= cx1[None, :]) & (py >= cy1[None, :]) &
            (px <= cx2[None, :]) & (py <= cy2[None, :])
        )

        max_reg = ltrb.amax(dim=2)
        in_range = (max_reg >= reg_range[0]) & (max_reg <= reg_range[1])
        valid = inside_box & inside_center & in_range

        areas = ((x2 - x1) * (y2 - y1))[None, :].expand(n, -1).clone()
        areas[~valid] = float("inf")
        min_area, matched = areas.min(dim=1)
        positive = torch.isfinite(min_area)
        if positive.any():
            matched_pos = matched[positive]
            reg_targets[positive] = ltrb[positive, matched_pos]
            cls_targets[positive, labels[matched_pos]] = 1.0
        return cls_targets, reg_targets, positive

    def forward(self, model, outputs, targets):
        levels = outputs["levels"]
        device = levels[0]["cls"].device
        batch = levels[0]["cls"].shape[0]
        total_cls = torch.zeros((), device=device)
        total_ctr = torch.zeros((), device=device)
        total_giou = torch.zeros((), device=device)
        total_nwd = torch.zeros((), device=device)
        total_pos = 0

        for stride, reg_range, pred in zip(self.strides, self.ranges, levels):
            cls_logits = pred["cls"].permute(0, 2, 3, 1).reshape(batch, -1, self.num_classes)
            ctr_logits = pred["ctr"].permute(0, 2, 3, 1).reshape(batch, -1)
            reg_raw = pred["reg"].permute(0, 2, 3, 1).reshape(batch, -1, 4)
            h, w = pred["cls"].shape[-2:]
            points = model.make_points(h, w, stride, device, cls_logits.dtype)

            for bi in range(batch):
                boxes = targets[bi]["boxes"].to(device)
                labels = targets[bi]["labels"].to(device)
                cls_t, reg_t, pos = self.assign_level(points, boxes, labels, stride, reg_range)
                npos = int(pos.sum().item())
                total_pos += npos
                total_cls = total_cls + sigmoid_focal_loss(cls_logits[bi], cls_t)

                if npos == 0:
                    continue

                ctr_t = centerness_target(reg_t[pos])
                total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                    ctr_logits[bi][pos], ctr_t, reduction="sum"
                )

                reg_p = F.softplus(reg_raw[bi][pos]) * stride
                p = points[pos]
                pred_boxes = torch.stack(
                    [p[:, 0] - reg_p[:, 0], p[:, 1] - reg_p[:, 1],
                     p[:, 0] + reg_p[:, 2], p[:, 1] + reg_p[:, 3]], dim=1
                )
                gt_boxes = torch.stack(
                    [p[:, 0] - reg_t[pos, 0], p[:, 1] - reg_t[pos, 1],
                     p[:, 0] + reg_t[pos, 2], p[:, 1] + reg_t[pos, 3]], dim=1
                )
                weight = ctr_t.detach()
                total_giou = total_giou + (
                    (1.0 - aligned_giou(pred_boxes, gt_boxes)) * weight
                ).sum()
                total_nwd = total_nwd + (
                    (1.0 - nwd_similarity(pred_boxes, gt_boxes, self.nwd_constant)) * weight
                ).sum()

        normalizer = max(1, total_pos)
        cls_loss = total_cls / normalizer
        ctr_loss = total_ctr / normalizer
        giou_loss = total_giou / normalizer
        nwd_loss = total_nwd / normalizer
        box_loss = self.giou_weight * giou_loss + self.nwd_weight * nwd_loss
        align_loss = torch.zeros((), device=device)
        aux = outputs.get("alignment_aux", {}) if isinstance(outputs, dict) else {}
        for flow in aux.get("flows", []):
            if flow is None:
                continue
            smooth = torch.zeros((), device=device)
            if flow.shape[-1] > 1:
                smooth = smooth + (flow[..., 1:] - flow[..., :-1]).abs().mean()
            if flow.shape[-2] > 1:
                smooth = smooth + (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
            align_loss = align_loss + self.flow_smooth_weight * smooth
            align_loss = align_loss + self.flow_magnitude_weight * flow.abs().mean()
        for affine in outputs.get("affine_params", []):
            if affine is None or affine.numel() == 0:
                continue
            tx_ty = affine[:, :2]
            scale = affine[:, 2:4]
            align_loss = align_loss + self.affine_weight * (tx_ty.abs().mean() + (scale - 1).abs().mean())

        total = (
            self.cls_weight * cls_loss +
            self.ctr_weight * ctr_loss +
            self.box_weight * box_loss +
            self.align_weight * align_loss
        )
        return {
            "loss": total,
            "loss_cls": cls_loss.detach(),
            "loss_ctr": ctr_loss.detach(),
            "loss_giou": giou_loss.detach(),
            "loss_nwd": nwd_loss.detach(),
            "loss_align": align_loss.detach(),
            "num_pos": torch.tensor(float(total_pos), device=device),
        }
