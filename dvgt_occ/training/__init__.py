"""Training utilities for DVGT-Occ stage-B overfit and beyond."""

from .batch import (
    DEFAULT_CACHE_KEYS,
    DEFAULT_SUPERVISION_KEYS,
    collate_dvgt_occ_batch,
    move_batch_to_device,
)
from .loss_builder import DVGTOccLossBuilder
from .metrics import binary_iou_from_logits, reduce_metrics
from .stage_scheduler import (
    DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS,
    LossWeights,
    resolve_loss_weights,
    stage_b_after_stability_weights,
)

__all__ = [
    "DEFAULT_CACHE_KEYS",
    "DEFAULT_SUPERVISION_KEYS",
    "DVGTOccLossBuilder",
    "DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS",
    "LossWeights",
    "resolve_loss_weights",
    "stage_b_after_stability_weights",
    "binary_iou_from_logits",
    "reduce_metrics",
    "collate_dvgt_occ_batch",
    "move_batch_to_device",
]
