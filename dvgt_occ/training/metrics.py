"""Validation metrics for Stage-C/D training."""

from __future__ import annotations

from typing import Dict

import torch


def binary_iou_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = torch.sigmoid(logits) > threshold
    tgt = target > 0.5
    inter = (pred & tgt).float().sum()
    union = (pred | tgt).float().sum()
    if union <= 0:
        return logits.new_tensor(1.0)
    return inter / union


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
