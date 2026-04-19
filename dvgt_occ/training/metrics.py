"""Validation metrics for Stage-C/D training."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def _align_binary_tensors(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    tgt = target
    valid = valid_mask
    while tgt.ndim < logits.ndim:
        tgt = tgt.unsqueeze(2)
    while valid is not None and valid.ndim < logits.ndim:
        valid = valid.unsqueeze(2)
    return tgt, valid


def binary_iou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    target, valid_mask = _align_binary_tensors(logits, target, valid_mask)
    pred = torch.sigmoid(logits) > threshold
    tgt = target > 0.5
    if valid_mask is not None:
        valid = valid_mask > 0.5
        if float(valid.sum().item()) <= 0:
            return logits.new_tensor(0.0)
        pred = pred & valid
        tgt = tgt & valid
    inter = (pred & tgt).float().sum()
    union = (pred | tgt).float().sum()
    if union <= 0:
        return logits.new_tensor(0.0)
    return inter / union


def soft_iou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    target, valid_mask = _align_binary_tensors(logits, target, valid_mask)
    prob = torch.sigmoid(logits)
    tgt = target.float()
    if valid_mask is not None:
        valid = (valid_mask > 0.5).float()
        if float(valid.sum().item()) <= 0:
            return logits.new_tensor(0.0)
        prob = prob * valid
        tgt = tgt * valid
    inter = (prob * tgt).sum()
    union = (prob + tgt - prob * tgt).sum()
    if float(union.item()) <= 0:
        return logits.new_tensor(0.0)
    return (inter + eps) / (union + eps)


def iou_threshold_sweep_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> Dict[str, torch.Tensor]:
    ious = []
    thresh_tensors = []
    for threshold in thresholds:
        iou = binary_iou_from_logits(logits, target, threshold=threshold, valid_mask=valid_mask)
        ious.append(iou)
        thresh_tensors.append(logits.new_tensor(float(threshold), dtype=torch.float32))
    iou_stack = torch.stack(ious)
    thresh_stack = torch.stack(thresh_tensors)
    best_idx = int(torch.argmax(iou_stack).item())
    results: Dict[str, torch.Tensor] = {
        "best_iou": iou_stack[best_idx],
        "best_threshold": thresh_stack[best_idx],
    }
    for threshold, value in zip(thresholds, ious):
        key = f"iou_t{int(round(float(threshold) * 100)):02d}"
        results[key] = value
    return results


def binary_stats_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    valid_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    target, valid_mask = _align_binary_tensors(logits, target, valid_mask)
    probs = torch.sigmoid(logits)
    pred = probs > threshold
    tgt = target > 0.5
    if valid_mask is not None:
        valid = valid_mask > 0.5
        valid_total = float(valid.sum().item())
        if valid_total <= 0:
            zero = logits.new_tensor(0.0, dtype=torch.float32)
            return {
                'precision': zero,
                'recall': zero,
                'dice': zero,
                'pred_pos_rate': zero,
                'target_pos_rate': zero,
                'prob_mean': zero,
                'valid_rate': zero,
            }
        pred = pred & valid
        tgt = tgt & valid
        probs = probs * valid.float()
        total = torch.tensor(valid_total, device=logits.device, dtype=torch.float32)
        valid_rate = total / torch.tensor(float(valid.numel()), device=logits.device, dtype=torch.float32)
    else:
        total = torch.tensor(float(pred.numel()), device=logits.device, dtype=torch.float32)
        valid_rate = torch.tensor(1.0, device=logits.device, dtype=torch.float32)

    tp = (pred & tgt).float().sum()
    fp = (pred & ~tgt).float().sum()
    fn = (~pred & tgt).float().sum()

    pred_pos = pred.float().sum()
    tgt_pos = tgt.float().sum()

    precision = tp / pred_pos.clamp_min(1.0)
    recall = tp / tgt_pos.clamp_min(1.0)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0)
    return {
        'precision': precision,
        'recall': recall,
        'dice': dice,
        'pred_pos_rate': pred_pos / total.clamp_min(1.0),
        'target_pos_rate': tgt_pos / total.clamp_min(1.0),
        'prob_mean': probs.float().sum() / total.clamp_min(1.0),
        'valid_rate': valid_rate,
    }


def render_valid_mask_from_points_conf(points_conf: torch.Tensor, threshold: float = 0.30, dilate_px: int = 5) -> torch.Tensor:
    b, t, v, h, w = points_conf.shape
    mask = (points_conf > threshold).float().reshape(b * t * v, 1, h, w)
    if dilate_px > 0:
        kernel = dilate_px * 2 + 1
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilate_px)
    return mask.reshape(b, t, v, 1, h, w)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.float()
    denom = valid.sum().clamp_min(1.0)
    return ((pred - target).abs() * valid).sum() / denom


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    valid = valid_mask.float()
    mse = (((pred - target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)
    return -10.0 * torch.log10(mse.clamp_min(eps))


def reduce_metrics(metrics: Dict[str, torch.Tensor], ddp_enabled: bool) -> Dict[str, float]:
    if not ddp_enabled:
        return {key: float(value.detach().item()) for key, value in metrics.items()}
    import torch.distributed as dist

    reduced: Dict[str, float] = {}
    for key, value in metrics.items():
        scalar = value.detach().clone()
        dist.all_reduce(scalar, op=dist.ReduceOp.AVG)
        reduced[key] = float(scalar.item())
    return reduced
