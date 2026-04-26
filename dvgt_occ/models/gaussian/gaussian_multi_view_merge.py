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
        dense_feat = gaussians.dense_feat.clone()
        offset = gaussians.offset.clone()
        opacity = gaussians.opacity.clone()
        scale = gaussians.scale.clone()
        rotation = gaussians.rotation.clone()
        feat_dc = gaussians.feat_dc.clone()
        keep_score = gaussians.keep_score.clone()
        instance_affinity = gaussians.instance_affinity.clone()
        motion_code = gaussians.motion_code.clone()
        dynamic_logit = gaussians.dynamic_logit.clone() if gaussians.dynamic_logit is not None else None

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
                    center_vals = center[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    offset_vals = offset[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    opacity_vals = opacity[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    scale_vals = scale[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    rotation_vals = rotation[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    feat_dc_vals = feat_dc[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    keep_vals = keep_score[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    dense_vals = dense_feat[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    instance_vals = instance_affinity[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    motion_vals = motion_code[batch_idx, time_idx, sel_v, sel_h, sel_w]
                    dynamic_vals = dynamic_logit[batch_idx, time_idx, sel_v, sel_h, sel_w] if dynamic_logit is not None else None

                    weight_view = weights[:, None]
                    mean_center = (center_vals * weight_view).sum(dim=0).to(center.dtype)
                    mean_offset = (offset_vals * weight_view).sum(dim=0).to(offset.dtype)
                    mean_opacity = (opacity_vals * weight_view).sum(dim=0).to(opacity.dtype)
                    mean_scale = (scale_vals * weight_view).sum(dim=0).to(scale.dtype)
                    mean_rotation = F.normalize((rotation_vals * weight_view).sum(dim=0), dim=-1).to(rotation.dtype)
                    mean_feat_dc = (feat_dc_vals * weight_view).sum(dim=0).to(feat_dc.dtype)
                    mean_keep = keep_vals.max(dim=0).values.to(keep_score.dtype)
                    mean_dense = (dense_vals * weight_view).sum(dim=0).to(dense_feat.dtype)
                    mean_instance = (instance_vals * weight_view).sum(dim=0).to(instance_affinity.dtype)
                    mean_motion = (motion_vals * weight_view).sum(dim=0).to(motion_code.dtype)
                    mean_dynamic = (
                        (dynamic_vals * weight_view).sum(dim=0).to(dynamic_logit.dtype)
                        if dynamic_vals is not None and dynamic_logit is not None
                        else None
                    )

                    center[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_center
                    dense_feat[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_dense
                    offset[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_offset
                    opacity[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_opacity
                    scale[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_scale
                    rotation[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_rotation
                    feat_dc[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_feat_dc
                    keep_score[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_keep
                    instance_affinity[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_instance
                    motion_code[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_motion
                    if dynamic_logit is not None and mean_dynamic is not None:
                        dynamic_logit[batch_idx, time_idx, rep_v, rep_h, rep_w] = mean_dynamic

                    dup_mask = torch.ones_like(sel_v, dtype=torch.bool)
                    dup_mask[rep_idx] = False
                    if torch.any(dup_mask):
                        dup_v = sel_v[dup_mask]
                        dup_h = sel_h[dup_mask]
                        dup_w = sel_w[dup_mask]
                        center[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_center
                        dense_feat[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_dense
                        offset[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_offset
                        opacity[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_opacity * self.duplicate_keep_scale
                        scale[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_scale
                        rotation[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_rotation
                        feat_dc[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_feat_dc
                        keep_score[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_keep * self.duplicate_keep_scale
                        instance_affinity[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_instance
                        motion_code[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_motion
                        if dynamic_logit is not None and mean_dynamic is not None:
                            dynamic_logit[batch_idx, time_idx, dup_v, dup_h, dup_w] = mean_dynamic

        return GaussianOutput(
            dense_feat=dense_feat,
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
            dynamic_logit=dynamic_logit,
        )
