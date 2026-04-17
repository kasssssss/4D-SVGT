"""2D-to-3D lift interface for first-frame ego occupancy volumes."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, feat_1_4: torch.Tensor, points: torch.Tensor, points_conf: torch.Tensor) -> torch.Tensor:
        b, t, v, c, h4, w4 = feat_1_4.shape
        nz, ny, nx = self.occ_shape_zyx
        num_voxels = nz * ny * nx

        feat_flat = feat_1_4.permute(0, 1, 2, 4, 5, 3).reshape(b * t, v * h4 * w4, c).float()

        points_ds = self._resize_points(points, size=(h4, w4))
        conf_ds = self._resize_conf(points_conf, size=(h4, w4))
        points_flat = points_ds.permute(0, 1, 2, 4, 5, 3).reshape(b * t, v * h4 * w4, 3)
        conf_flat = conf_ds.reshape(b * t, v * h4 * w4)

        x = points_flat[..., 0]
        y = points_flat[..., 1]
        z = points_flat[..., 2]

        valid = torch.isfinite(points_flat).all(dim=-1)
        valid &= conf_flat > 1e-4
        valid &= (x >= self.x_range[0]) & (x < self.x_range[1])
        valid &= (y >= self.y_range[0]) & (y < self.y_range[1])
        valid &= (z >= self.z_range[0]) & (z < self.z_range[1])

        ix = torch.floor((x - self.x_range[0]) / self.voxel_size).long().clamp_(0, nx - 1)
        iy = torch.floor((y - self.y_range[0]) / self.voxel_size).long().clamp_(0, ny - 1)
        iz = torch.floor((z - self.z_range[0]) / self.voxel_size).long().clamp_(0, nz - 1)
        linear = iz * (ny * nx) + iy * nx + ix
        linear = torch.where(valid, linear, torch.zeros_like(linear))

        weighted_feat = feat_flat * conf_flat.unsqueeze(-1) * valid.unsqueeze(-1)
        idx_feat = linear.unsqueeze(-1).expand(-1, -1, c)

        volume_feat = feat_flat.new_zeros((b * t, num_voxels, c), dtype=torch.float32)
        volume_weight = feat_flat.new_zeros((b * t, num_voxels, 1), dtype=torch.float32)
        volume_feat.scatter_add_(1, idx_feat, weighted_feat)
        volume_weight.scatter_add_(1, linear.unsqueeze(-1), conf_flat.unsqueeze(-1) * valid.unsqueeze(-1))

        volume = volume_feat / volume_weight.clamp_min(1e-6)
        volume = volume.reshape(b, t, nz, ny, nx, c).permute(0, 1, 5, 2, 3, 4).contiguous()
        volume = volume.to(feat_1_4.dtype)
        return self.proj(volume.reshape(b * t, c, nz, ny, nx)).reshape(b, t, c, nz, ny, nx)

    @staticmethod
    def _resize_points(points: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        b, t, v, h, w, c = points.shape
        x = points.permute(0, 1, 2, 5, 3, 4).reshape(b * t * v, c, h, w)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x.reshape(b, t, v, c, size[0], size[1]).permute(0, 1, 2, 4, 5, 3).contiguous()

    @staticmethod
    def _resize_conf(points_conf: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        b, t, v, h, w = points_conf.shape
        x = points_conf.reshape(b * t * v, 1, h, w)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x.reshape(b, t, v, size[0], size[1]).contiguous()
