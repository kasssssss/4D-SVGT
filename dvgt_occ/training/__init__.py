"""Training utilities for DVGT-Occ stage-B overfit and beyond."""

from .batch import (
    DEFAULT_CACHE_KEYS,
    DEFAULT_SUPERVISION_KEYS,
    collate_dvgt_occ_batch,
    move_batch_to_device,
)
from .loss_builder import DVGTOccLossBuilder
from .metrics import (
    binary_iou_from_logits,
    binary_stats_from_logits,
    iou_threshold_sweep_from_logits,
    masked_l1,
    masked_psnr,
    reduce_metrics,
    render_valid_mask_from_points_conf,
    soft_iou_from_logits,
)
from .stage_scheduler import (
    DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS,
    LossWeights,
    resolve_loss_weights,
    stage_b_after_stability_weights,
    stage_c_bridge_warmup_weights,
)
from .visualization import save_training_visualization

__all__ = [
    "DEFAULT_CACHE_KEYS",
    "DEFAULT_SUPERVISION_KEYS",
    "DVGTOccLossBuilder",
    "DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS",
    "LossWeights",
    "resolve_loss_weights",
    "stage_b_after_stability_weights",
    "stage_c_bridge_warmup_weights",
    "binary_iou_from_logits",
    "binary_stats_from_logits",
    "iou_threshold_sweep_from_logits",
    "soft_iou_from_logits",
    "masked_l1",
    "masked_psnr",
    "render_valid_mask_from_points_conf",
    "reduce_metrics",
    "save_training_visualization",
    "collate_dvgt_occ_batch",
    "move_batch_to_device",
]
