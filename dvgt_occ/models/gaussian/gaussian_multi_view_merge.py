"""Cross-view Gaussian dedup / merge scaffold in ego0 coordinates."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dvgt_occ.types import GaussianOutput


class GaussianMultiViewMerge(nn.Module):
    """Deduplicate track-aligned view candidates into a more entity-like set.

    The scaffold still keeps the dense tensor shape for downstream modules, but
    once multiple cells share the same global track id we now pick a single
    representative carrier and strongly suppress the rest instead of simply
    averaging back into every original cell.
    """

    def __init__(self, distance_threshold: float = 2.0, duplicate_keep_scale: float = 0.05) -> None:
        super().__init__()
        self.distance_threshold = distance_threshold
        self.duplicate_keep_scale = duplicate_keep_scale

    def forward(self, gaussians: GaussianOutput, global_track_id: torch.Tensor | None = None) -> GaussianOutput:
        if global_track_id is None:
            return gaussians

        center = gaussians.center.clone()
        offset = gaussians.offset.clone()
        opacity = gaussians.opacity.clone()
        scale = gaussians.scale.clone()
        rotation = gaussians.rotation.clone()
        feat_dc = gaussians.feat_dc.clone()
        keep_score = gaussians.keep_score.clone()
        instance_affinity = gaussians.instance_affinity.clone()
        motion_code = gaussians.motion_code.clone()

        b, t, v, h, w = global_track_id.shape
        for batch_idx in range(b):
            for time_idx in range(t):
                ids = global_track_id[batch_idx, time_idx]
                valid_ids = torch.unique(ids[ids >= 0])
                for track_id in valid_ids.tolist():
                    mask = ids == track_id
                    if int(mask.sum().item()) < 2:
                        continue
                    coords = torch.nonzero(mask, as_tuple=False)
                    sel_v, sel_h, sel_w = coords[:, 0], coords[:, 1], coords[:, 2]
                    centers = center[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    keep_vals = keep_score[batch_idx, time_idx, sel_v, sel_h, sel_w, 0]
                    weights = keep_vals.softmax(dim=0)
                    mean_center = (centers * weights[:, None]).sum(dim=0)
                    distances = torch.linalg.norm(centers - mean_center[None, :], dim=-1)
                    active = distances <= self.distance_threshold
                    if int(active.sum().item()) < 1:
                        continue
                    sel_v = sel_v[active]
                    sel_h = sel_h[active]
                    sel_w = sel_w[active]
                    weights = keep_score[batch_idx, time_idx, sel_v, sel_h, sel_w, 0].softmax(dim=0)
                    rep_idx = int(torch.argmax(weights).item())
                    rep_v = int(sel_v[rep_idx].item())
                    rep_h = int(sel_h[rep_idx].item())
                    rep_w = int(sel_w[rep_idx].item())

                    def weighted_mean(tensor: torch.Tensor) -> torch.Tensor:
                        values = tensor[batch_idx, time_idx, sel_v, sel_h, sel_w]
                        return (values * weights[:, None]).sum(dim=0)

                    mean_center = weighted_mean(center)
                    mean_offset = weighted_mean(offset)
                    mean_opacity = weighted_mean(opacity)
                    mean_scale = weighted_mean(scale)
                    mean_rotation = F.normalize(weighted_mean(rotation), dim=-1)
                    mean_feat_dc = weighted_mean(feat_dc)
                    mean_keep = keep_score[batch_idx, time_idx, sel_v, sel_h, sel_w].max(dim=0).values
                    mean_instance = weighted_mean(instance_affinity)
                    mean_motion = weighted_mean(motion_code)

                    center[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_center
                    offset[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_offset
                    opacity[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_opacity
                    scale[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_scale
                    rotation[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_rotation
                    feat_dc[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_feat_dc
                    keep_score[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_keep
                    instance_affinity[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_instance
                    motion_code[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_motion

                    for dup in range(sel_v.shape[0]):
                        if dup == rep_idx:
                            continue
                        dv = int(sel_v[dup].item())
                        dh = int(sel_h[dup].item())
                        dw = int(sel_w[dup].item())
                        center[batch_idx, time_idx, dv, dh, dw] = mean_center
                        offset[batch_idx, time_idx, dv, dh, dw] = mean_offset
                        opacity[batch_idx, time_idx, dv, dh, dw] = mean_opacity * self.duplicate_keep_scale
                        scale[batch_idx, time_idx, dv, dh, dw] = mean_scale
                        rotation[batch_idx, time_idx, dv, dh, dw] = mean_rotation
                        feat_dc[batch_idx, time_idx, dv, dh, dw] = mean_feat_dc
                        keep_score[batch_idx, time_idx, dv, dh, dw] = mean_keep * self.duplicate_keep_scale
                        instance_affinity[batch_idx, time_idx, dv, dh, dw] = mean_instance
                        motion_code[batch_idx, time_idx, dv, dh, dw] = mean_motion

        return GaussianOutput(
            center=center,
            offset=offset,
            opacity=opacity,
            scale=scale,
            rotation=rotation,
            feat_dc=feat_dc,
            keep_score=keep_score,
            instance_affinity=instance_affinity,
            motion_code=motion_code,
            aux_decoder_full=gaussians.aux_decoder_full,
        )
