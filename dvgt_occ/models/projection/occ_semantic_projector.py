"""Project occupancy semantics to 2D training resolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccSemanticProjector(nn.Module):
    def __init__(
        self,
        semantic_classes: int = 18,
        projected_classes: int = 8,
        output_size: tuple[int, int] = (224, 448),
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.projected_classes = projected_classes
        self.subset_proj = nn.Conv3d(semantic_classes, projected_classes, kernel_size=1)

    def forward(self, occ_logit: torch.Tensor, sem_logit: torch.Tensor, num_views: int) -> torch.Tensor:
        b, t, _, nz, ny, nx = occ_logit.shape
        density = torch.sigmoid(occ_logit.squeeze(2))
        sem_subset = self.subset_proj(sem_logit.reshape(b * t, sem_logit.shape[2], nz, ny, nx)).reshape(
            b, t, self.projected_classes, nz, ny, nx
        )
        alpha = density / density.sum(dim=2, keepdim=True).clamp_min(1e-6)
        sem_xy = (sem_subset * alpha.unsqueeze(2)).sum(dim=3)
        sem_xy = F.interpolate(
            sem_xy.reshape(b * t, self.projected_classes, ny, nx),
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, self.projected_classes, self.output_size[0], self.output_size[1])
        return sem_xy.unsqueeze(2).expand(-1, -1, num_views, -1, -1, -1).contiguous()
