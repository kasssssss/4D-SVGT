"""Query-to-Gaussian routing with an explicit null/background branch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dvgt_occ.types import DynamicQueryOutput, GaussianAssignmentOutput, GaussianOutput


class GaussianQueryAssignment(nn.Module):
    def __init__(
        self,
        query_dim: int = 256,
        gaussian_feat_dim: int = 48,
        routing_dim: int = 128,
        dilation: float = 1.5,
        min_extent: float = 0.5,
        gate_temperature: float = 3.0,
        feature_temperature: float = 6.0,
        keep_score_weight: float = 0.5,
        background_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.min_extent = min_extent
        self.gate_temperature = gate_temperature
        self.feature_temperature = feature_temperature
        self.keep_score_weight = keep_score_weight
        self.query_proj = nn.Linear(query_dim, routing_dim)
        self.gaussian_proj = nn.Linear(gaussian_feat_dim, routing_dim)
        self.background_bias = nn.Parameter(torch.tensor(float(background_bias)))

    def forward(self, gaussians: GaussianOutput, queries: DynamicQueryOutput) -> GaussianAssignmentOutput:
        b, t = gaussians.center.shape[:2]
        centers = gaussians.center.reshape(b, t, -1, 3)
        query_boxes = queries.query_box3d[..., :7]
        query_centers = query_boxes[..., :3]
        query_sizes = query_boxes[..., 3:6].abs().clamp_min(self.min_extent)

        delta = (centers[:, :, :, None, :] - query_centers[:, :, None, :, :]).abs()
        normalized = delta / (0.5 * self.dilation * query_sizes[:, :, None, :, :])
        signed_distance = normalized.max(dim=-1).values - 1.0
        local_gate = (signed_distance <= 0.0).float()
        gate_logit = (-signed_distance * self.gate_temperature).clamp(min=-8.0, max=8.0)

        gaussian_feat = torch.cat([gaussians.instance_affinity, gaussians.motion_code], dim=-1)
        feat_dim = gaussian_feat.shape[-1]
        gaussian_feat = gaussian_feat.reshape(b, t, -1, feat_dim)
        query_feat = F.normalize(self.query_proj(queries.query_feat), dim=-1)
        gaussian_feat = F.normalize(self.gaussian_proj(gaussian_feat), dim=-1)
        feature_similarity = torch.matmul(gaussian_feat, query_feat.transpose(-1, -2))

        presence = queries.presence_logit[:, :, None, :]
        routing_keep = torch.logit(gaussians.keep_score.reshape(b, t, -1, 1).clamp(1e-4, 1.0 - 1.0e-4))
        fg_logits = (
            presence
            + self.feature_temperature * feature_similarity
            + gate_logit
            + self.keep_score_weight * routing_keep
        )

        bg_logits = torch.zeros_like(fg_logits[..., :1]) + self.background_bias
        logits = torch.cat([bg_logits, fg_logits], dim=-1)
        probs = torch.softmax(logits, dim=-1)

        return GaussianAssignmentOutput(
            assignment_prob=probs.reshape(b, -1, probs.shape[-1]),
            assigned_query=torch.argmax(probs, dim=-1).reshape(b, -1) - 1,
            background_prob=probs[..., 0].reshape(b, -1),
            local_gate=local_gate.reshape(b, -1, local_gate.shape[-1]),
            foreground_logit=fg_logits.reshape(b, -1, fg_logits.shape[-1]),
            background_logit=bg_logits.reshape(b, -1, 1),
            feature_similarity=feature_similarity.reshape(b, -1, feature_similarity.shape[-1]),
            routing_keep_score=gaussians.keep_score.reshape(b, -1, 1),
        )
