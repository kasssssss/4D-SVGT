"""Lightweight differentiable mask renderer over the per-view Gaussian grid."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dvgt_occ.types import GaussianAssignmentOutput, GaussianOutput, RenderOutput


class GaussianMaskRenderer(nn.Module):
    def __init__(self, output_size: tuple[int, int] = (224, 448)) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(
        self,
        gaussians: GaussianOutput,
        assignment: GaussianAssignmentOutput,
        sem_proj_2d: torch.Tensor,
    ) -> RenderOutput:
        b, t, v, h, w = gaussians.opacity.shape[:5]
        opacity = gaussians.opacity.squeeze(-1)
        background = assignment.background_prob.reshape(b, t, v, h, w)
        dynamicness = 1.0 - background

        render_dynamic = self._upsample(opacity * dynamicness)
        render_static = self._upsample(opacity * background)
        render_all = self._upsample(opacity)
        return RenderOutput(
            render_alpha_static=render_static.unsqueeze(3),
            render_alpha_dynamic=render_dynamic.unsqueeze(3),
            render_alpha_all=render_all.unsqueeze(3),
            sem_proj_2d=sem_proj_2d,
        )

    def _upsample(self, alpha: torch.Tensor) -> torch.Tensor:
        b, t, v, h, w = alpha.shape
        up = F.interpolate(
            alpha.reshape(b * t * v, 1, h, w),
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        return up.reshape(b, t, v, self.output_size[0], self.output_size[1])
