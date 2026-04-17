"""Query-to-Gaussian routing with an explicit null/background branch."""

from __future__ import annotations

import torch
import torch.nn as nn

from dvgt_occ.types import DynamicQueryOutput, GaussianAssignmentOutput, GaussianOutput


class GaussianQueryAssignment(nn.Module):
    def __init__(self, dilation: float = 1.5, min_extent: float = 0.5) -> None:
        super().__init__()
        self.dilation = dilation
        self.min_extent = min_extent

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

        presence = queries.presence_logit[:, :, None, :]
        fg_logits = presence - signed_distance.clamp_min(0.0) * 4.0
        fg_logits = fg_logits.masked_fill(local_gate <= 0.0, -1e4)

        bg_logits = torch.zeros_like(fg_logits[..., :1]) + (1.0 - gaussians.confidence.reshape(b, t, -1, 1))
        logits = torch.cat([bg_logits, fg_logits], dim=-1)
        probs = torch.softmax(logits, dim=-1)

        return GaussianAssignmentOutput(
            assignment_prob=probs.reshape(b, -1, probs.shape[-1]),
            assigned_query=torch.argmax(probs, dim=-1).reshape(b, -1) - 1,
            background_prob=probs[..., 0].reshape(b, -1),
            local_gate=local_gate.reshape(b, -1, local_gate.shape[-1]),
        )
