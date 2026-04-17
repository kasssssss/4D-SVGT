"""Local Voxel-Gaussian hard bridge utilities."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def point_voxel_hash_mean(
    points_xyz: torch.Tensor,
    values: torch.Tensor,
    grid_min_xyz: Tuple[float, float, float] = (-40.0, -40.0, -2.0),
    voxel_size: float = 0.8,
    grid_shape_zyx: Tuple[int, int, int] = (10, 100, 100),
) -> torch.Tensor:
    """Aggregate point features into voxels with a hash mean.

    Args:
        points_xyz: ``[B, N, 3]`` in first-frame ego coordinates.
        values: ``[B, N, C]`` point or Gaussian features.
    Returns:
        Dense volume ``[B, C, Nz, Ny, Nx]``.
    """

    if points_xyz.ndim != 3 or values.ndim != 3:
        raise ValueError("Expected points_xyz [B,N,3] and values [B,N,C]")
    b, n, _ = points_xyz.shape
    c = values.shape[-1]
    nz, ny, nx = grid_shape_zyx
    device = values.device
    mins = torch.tensor(grid_min_xyz, device=device, dtype=points_xyz.dtype)
    ijk = torch.floor((points_xyz - mins) / voxel_size).long()
    ix, iy, iz = ijk[..., 0], ijk[..., 1], ijk[..., 2]
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    linear = iz * (ny * nx) + iy * nx + ix
    volume = torch.zeros(b, nz * ny * nx, c, device=device, dtype=values.dtype)
    counts = torch.zeros(b, nz * ny * nx, 1, device=device, dtype=values.dtype)
    for batch in range(b):
        idx = linear[batch, valid[batch]]
        if idx.numel() == 0:
            continue
        volume[batch].index_add_(0, idx, values[batch, valid[batch]])
        counts[batch].index_add_(0, idx, torch.ones(idx.numel(), 1, device=device, dtype=values.dtype))
    volume = volume / counts.clamp_min(1.0)
    return volume.reshape(b, nz, ny, nx, c).permute(0, 4, 1, 2, 3).contiguous()


class GSOccLocalBridge(nn.Module):
    def __init__(
        self,
        occ_channels: int = 256,
        gs_channels: int = 48,
        grid_min_xyz: Tuple[float, float, float] = (-40.0, -40.0, -2.0),
        voxel_size: float = 0.8,
        grid_shape_zyx: Tuple[int, int, int] = (10, 100, 100),
    ) -> None:
        super().__init__()
        self.grid_min_xyz = grid_min_xyz
        self.voxel_size = voxel_size
        self.grid_shape_zyx = grid_shape_zyx
        self.gs_to_occ = nn.Linear(gs_channels, occ_channels)
        self.occ_to_gs = nn.Linear(occ_channels, gs_channels)

    def forward(self, centers: torch.Tensor, gs_features: torch.Tensor, occ_volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, v, h, w, _ = centers.shape
        bt = b * t
        n = v * h * w
        flat_centers = centers.reshape(bt, n, 3)
        flat_features = gs_features.reshape(bt, n, gs_features.shape[-1])
        gs_to_occ_local = point_voxel_hash_mean(
            flat_centers,
            self.gs_to_occ(flat_features),
            grid_min_xyz=self.grid_min_xyz,
            voxel_size=self.voxel_size,
            grid_shape_zyx=self.grid_shape_zyx,
        )
        gs_to_occ_local = gs_to_occ_local.reshape(b, t, gs_to_occ_local.shape[1], *gs_to_occ_local.shape[-3:])

        occ_flat = occ_volume.reshape(bt, occ_volume.shape[2], *occ_volume.shape[-3:])
        grid = self._normalize_grid(flat_centers)
        sampled = F.grid_sample(
            occ_flat,
            grid.reshape(bt, n, 1, 1, 3),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(bt, occ_volume.shape[2], n).permute(0, 2, 1).contiguous()
        occ_to_gs_local = self.occ_to_gs(sampled).reshape(b, t, v, h, w, -1)
        return gs_to_occ_local, occ_to_gs_local

    def _normalize_grid(self, centers: torch.Tensor) -> torch.Tensor:
        mins = centers.new_tensor(self.grid_min_xyz)
        dims = centers.new_tensor(
            [
                self.grid_shape_zyx[2] * self.voxel_size,
                self.grid_shape_zyx[1] * self.voxel_size,
                self.grid_shape_zyx[0] * self.voxel_size,
            ]
        )
        xyz01 = (centers - mins) / dims.clamp_min(1e-6)
        grid_x = xyz01[..., 0] * 2.0 - 1.0
        grid_y = xyz01[..., 1] * 2.0 - 1.0
        grid_z = xyz01[..., 2] * 2.0 - 1.0
        return torch.stack([grid_x, grid_y, grid_z], dim=-1)
