"""Lightweight sky background model organized like DGGT's sky head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SkyRayBackground(nn.Module):
    """Predict RGB background from world ray directions.

    This is a lightweight stand-in for DGGT's sky head organization: an explicit
    background model trained with sky masks, separated from foreground GS
    rendering. We keep the implementation dependency-light so it plugs into the
    current DVGT stack cleanly.
    """

    def __init__(self, hidden_dim: int = 64, fourier_freqs: int = 6) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fourier_freqs = int(fourier_freqs)
        in_dim = 3 * (1 + 2 * self.fourier_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        camera_intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        h, w = int(image_size[0]), int(image_size[1])
        rays_world = self._world_rays(camera_intrinsics, camera_to_world, h, w)
        enc = self._encode_dir(rays_world)
        rgb = self.mlp(enc)
        return rgb.permute(0, 1, 2, 5, 3, 4).contiguous()

    def _encode_dir(self, direction: torch.Tensor) -> torch.Tensor:
        feats = [direction]
        for level in range(self.fourier_freqs):
            freq = float(2 ** level) * math.pi
            feats.append(torch.sin(direction * freq))
            feats.append(torch.cos(direction * freq))
        return torch.cat(feats, dim=-1)

    def _world_rays(
        self,
        camera_intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        device = camera_to_world.device
        dtype = camera_to_world.dtype
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        xs = xs + 0.5
        ys = ys + 0.5

        intr = camera_intrinsics
        if intr.dim() == 2:
            intr = intr.unsqueeze(0)
        if intr.dim() == 3:
            intr = intr.unsqueeze(1).expand(-1, camera_to_world.shape[1], -1, -1)

        fx = intr[..., 0][..., None, None]
        fy = intr[..., 1][..., None, None]
        cx = intr[..., 2][..., None, None]
        cy = intr[..., 3][..., None, None]

        x_cam = (xs - cx) / fx.clamp_min(1e-6)
        y_cam = (ys - cy) / fy.clamp_min(1e-6)
        z_cam = torch.ones_like(x_cam)
        rays_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        rot = camera_to_world[..., :3, :3]
        rays_world = torch.einsum("btvij,btvhwj->btvhwi", rot, rays_cam)
        rays_world = rays_world / rays_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return rays_world
