"""Multi-scale deformable 3D attention over occupancy volumes."""

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformable3DAttention(nn.Module):
    def __init__(
        self,
        query_dim: int = 256,
        samples_per_scale: int = 8,
        max_scales: int = 3,
        volume_channels: tuple[int, ...] = (64, 128, 128),
    ) -> None:
        super().__init__()
        self.samples_per_scale = samples_per_scale
        self.offset_heads = nn.ModuleList([nn.Linear(query_dim, samples_per_scale * 3) for _ in range(max_scales)])
        self.weight_heads = nn.ModuleList([nn.Linear(query_dim, samples_per_scale) for _ in range(max_scales)])
        self.value_projs = nn.ModuleList(
            [nn.Linear(volume_channels[idx] if idx < len(volume_channels) else query_dim, query_dim) for idx in range(max_scales)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(query_dim * max_scales),
            nn.Linear(query_dim * max_scales, query_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        reference_points: torch.Tensor,
        occ_volumes: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        sampled_per_scale = []
        scale_keys = sorted(occ_volumes.keys())
        for scale_idx, key in enumerate(scale_keys[: len(self.offset_heads)]):
            volume = occ_volumes[key]
            offsets = torch.tanh(self.offset_heads[scale_idx](queries)).view(
                queries.shape[0],
                queries.shape[1],
                self.samples_per_scale,
                3,
            ) * 0.25
            weights = torch.softmax(self.weight_heads[scale_idx](queries), dim=-1)
            grid = (reference_points.unsqueeze(2) + offsets).clamp(0.0, 1.0)
            grid = grid.mul(2.0).sub(1.0)
            grid = grid[:, :, :, None, :]
            sampled = F.grid_sample(
                volume,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.squeeze(-1).permute(0, 2, 3, 1).contiguous()
            sampled = (sampled * weights.unsqueeze(-1)).sum(dim=2)
            sampled = self.value_projs[scale_idx](sampled)
            sampled_per_scale.append(sampled)

        if len(sampled_per_scale) < len(self.offset_heads):
            sampled_per_scale.extend([torch.zeros_like(queries) for _ in range(len(self.offset_heads) - len(sampled_per_scale))])
        return self.out(torch.cat(sampled_per_scale, dim=-1))
