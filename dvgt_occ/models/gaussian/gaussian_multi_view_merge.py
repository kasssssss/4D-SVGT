"""Cross-view Gaussian dedup / merge scaffold in ego0 coordinates."""

from __future__ import annotations

import torch
import torch.nn as nn

from dvgt_occ.types import GaussianOutput


class GaussianMultiViewMerge(nn.Module):
    """Keep the scaffold honest about the v0.9.4 data/model contract.

    The real merge logic will cluster view-level Gaussian candidates in ego0
    space before the entity aggregator consumes them. For now this module keeps
    the API slot explicit so later training work does not accidentally wire the
    aggregator directly onto raw per-view candidates.
    """

    def __init__(self, distance_threshold: float = 2.0) -> None:
        super().__init__()
        self.distance_threshold = distance_threshold

    def forward(self, gaussians: GaussianOutput, global_track_id: torch.Tensor | None = None) -> GaussianOutput:
        if global_track_id is None:
            return gaussians

        center = gaussians.center.clone()
        offset = gaussians.offset.clone()
        opacity = gaussians.opacity.clone()
        scale = gaussians.scale.clone()
        rotation = gaussians.rotation.clone()
        feat_dc = gaussians.feat_dc.clone()
        confidence = gaussians.confidence.clone()
        instance_affinity = gaussians.instance_affinity.clone()
        motion_code = gaussians.motion_code.clone()

        b, t, v, h, w = global_track_id.shape
        for batch_idx in range(b):
            for time_idx in range(t):
                ids = global_track_id[batch_idx, time_idx]
                valid_ids = torch.unique(ids[ids >= 0])
                for track_id in valid_ids.tolist():
                    mask = ids == track_id
                    if mask.sum() < 2:
                        continue
                    center_bt = center[batch_idx, time_idx]
                    mean_center = center_bt[mask].mean(dim=0)
                    distance_ok = torch.linalg.norm(center_bt[mask] - mean_center, dim=-1) <= self.distance_threshold
                    if not torch.any(distance_ok):
                        continue
                    selected = torch.nonzero(mask, as_tuple=False)[distance_ok]
                    sel_v, sel_h, sel_w = selected[:, 0], selected[:, 1], selected[:, 2]
                    mean_center = center_bt[sel_v, sel_h, sel_w].mean(dim=0)
                    mean_offset = offset[batch_idx, time_idx, sel_v, sel_h, sel_w].mean(dim=0)
                    mean_opacity = opacity[batch_idx, time_idx, sel_v, sel_h, sel_w].mean(dim=0)
                    mean_scale = scale[batch_idx, time_idx, sel_v, sel_h, sel_w].mean(dim=0)
                    mean_rotation = torch.nn.functional.normalize(rotation[batch_idx, time_idx, sel_v, sel_h, sel_w].mean(dim=0), dim=-1)
                    mean_feat_dc = feat_dc[batch_idx, time_idx, sel_v, sel_h, sel_w].mean(dim=0)
                    mean_confidence = confidence[batch_idx, time_idx, sel_v, sel_h, sel_w].mean(dim=0)
                    mean_instance = instance_affinity[batch_idx, time_idx, sel_v, sel_h, sel_w].mean(dim=0)
                    mean_motion = motion_code[batch_idx, time_idx, sel_v, sel_h, sel_w].mean(dim=0)
                    center[batch_idx, time_idx, sel_v, sel_h, sel_w] = mean_center
                    offset[batch_idx, time_idx, sel_v, sel_h, sel_w] = mean_offset
                    opacity[batch_idx, time_idx, sel_v, sel_h, sel_w] = mean_opacity
                    scale[batch_idx, time_idx, sel_v, sel_h, sel_w] = mean_scale
                    rotation[batch_idx, time_idx, sel_v, sel_h, sel_w] = mean_rotation
                    feat_dc[batch_idx, time_idx, sel_v, sel_h, sel_w] = mean_feat_dc
                    confidence[batch_idx, time_idx, sel_v, sel_h, sel_w] = mean_confidence
                    instance_affinity[batch_idx, time_idx, sel_v, sel_h, sel_w] = mean_instance
                    motion_code[batch_idx, time_idx, sel_v, sel_h, sel_w] = mean_motion

        return GaussianOutput(
            center=center,
            offset=offset,
            opacity=opacity,
            scale=scale,
            rotation=rotation,
            feat_dc=feat_dc,
            confidence=confidence,
            instance_affinity=instance_affinity,
            motion_code=motion_code,
        )
