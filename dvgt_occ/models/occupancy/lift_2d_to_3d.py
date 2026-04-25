"""2D-to-3D lift interface for first-frame ego occupancy volumes."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class Lift2DTo3D(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        occ_shape_zyx: Tuple[int, int, int] = (10, 100, 100),
        x_range: Tuple[float, float] = (-40.0, 40.0),
        y_range: Tuple[float, float] = (-40.0, 40.0),
        z_range: Tuple[float, float] = (-2.0, 6.0),
        voxel_size: float = 0.8,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.occ_shape_zyx = occ_shape_zyx
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.voxel_size = voxel_size
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, feat_1_4: torch.Tensor, xyz_1_4: torch.Tensor, conf_1_4: torch.Tensor) -> torch.Tensor:
        b, t, v, c, h4, w4 = feat_1_4.shape
        nz, ny, nx = self.occ_shape_zyx
        num_voxels = nz * ny * nx

        feat_flat = feat_1_4.permute(0, 1, 2, 4, 5, 3).reshape(b * t, v * h4 * w4, c).float()
        points_flat = xyz_1_4.permute(0, 1, 2, 4, 5, 3).reshape(b * t, v * h4 * w4, 3).float()
        conf_flat = conf_1_4.reshape(b * t, v * h4 * w4).float()

        x = points_flat[..., 0]
        y = points_flat[..., 1]
        z = points_flat[..., 2]

        valid = torch.isfinite(points_flat).all(dim=-1)
        valid &= conf_flat > 1e-4
        valid &= (x >= self.x_range[0]) & (x < self.x_range[1])
        valid &= (y >= self.y_range[0]) & (y < self.y_range[1])
        valid &= (z >= self.z_range[0]) & (z < self.z_range[1])

        x_idx = (x - self.x_range[0]) / self.voxel_size
        y_idx = (y - self.y_range[0]) / self.voxel_size
        z_idx = (z - self.z_range[0]) / self.voxel_size
        x0 = torch.floor(x_idx)
        y0 = torch.floor(y_idx)
        z0 = torch.floor(z_idx)
        fx = (x_idx - x0).clamp(0.0, 1.0)
        fy = (y_idx - y0).clamp(0.0, 1.0)
        fz = (z_idx - z0).clamp(0.0, 1.0)

        volume_feat = feat_flat.new_zeros((b * t, num_voxels, c), dtype=torch.float32)
        volume_weight = feat_flat.new_zeros((b * t, num_voxels, 1), dtype=torch.float32)

        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix = (x0.long() + dx).clamp_(0, nx - 1)
                    iy = (y0.long() + dy).clamp_(0, ny - 1)
                    iz = (z0.long() + dz).clamp_(0, nz - 1)
                    neighbor_valid = valid
                    neighbor_valid &= (x0.long() + dx >= 0) & (x0.long() + dx < nx)
                    neighbor_valid &= (y0.long() + dy >= 0) & (y0.long() + dy < ny)
                    neighbor_valid &= (z0.long() + dz >= 0) & (z0.long() + dz < nz)
                    wx = fx if dx == 1 else (1.0 - fx)
                    wy = fy if dy == 1 else (1.0 - fy)
                    wz = fz if dz == 1 else (1.0 - fz)
                    weight = (wx * wy * wz * conf_flat * neighbor_valid).unsqueeze(-1)
                    linear = iz * (ny * nx) + iy * nx + ix
                    idx_feat = linear.unsqueeze(-1).expand(-1, -1, c)
                    volume_feat.scatter_add_(1, idx_feat, feat_flat * weight)
                    volume_weight.scatter_add_(1, linear.unsqueeze(-1), weight)

        volume = volume_feat / volume_weight.clamp_min(1e-6)
        volume = volume.reshape(b, t, nz, ny, nx, c).permute(0, 1, 5, 2, 3, 4).contiguous()
        volume = volume.to(feat_1_4.dtype)
        return self.proj(volume.reshape(b * t, c, nz, ny, nx)).reshape(b, t, c, nz, ny, nx)
