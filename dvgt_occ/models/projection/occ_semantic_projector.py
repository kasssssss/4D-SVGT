"""Camera-aware 3D occupancy semantic projection to 2D."""

from __future__ import annotations

import torch
import torch.nn as nn


class OccSemanticProjector(nn.Module):
    def __init__(
        self,
        semantic_classes: int = 18,
        projected_classes: int = 8,
        output_size: tuple[int, int] = (224, 448),
        x_range: tuple[float, float] = (-40.0, 40.0),
        y_range: tuple[float, float] = (-40.0, 40.0),
        z_range: tuple[float, float] = (-2.0, 6.0),
        voxel_size: float = 0.8,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.projected_classes = projected_classes
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.voxel_size = float(voxel_size)
        self.subset_proj = nn.Conv3d(semantic_classes, projected_classes, kernel_size=1)

    def forward(
        self,
        occ_logit: torch.Tensor,
        sem_logit: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        first_ego_pose_world: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _, nz, ny, nx = occ_logit.shape
        height, width = self.output_size
        sem_subset = self.subset_proj(sem_logit.reshape(b * t, sem_logit.shape[2], nz, ny, nx)).reshape(
            b, t, self.projected_classes, nz, ny, nx
        )
        density = torch.nn.functional.softplus(occ_logit.squeeze(2))
        alpha = 1.0 - torch.exp(-density)
        sem_prob = torch.softmax(sem_subset, dim=2)

        centers = self._voxel_centers(nx=nx, ny=ny, nz=nz, device=occ_logit.device, dtype=occ_logit.dtype)
        centers_h = torch.cat([centers, torch.ones((centers.shape[0], 1), device=centers.device, dtype=centers.dtype)], dim=-1)
        out = torch.zeros((b, t, camera_to_world.shape[2], self.projected_classes, height, width), device=occ_logit.device, dtype=occ_logit.dtype)

        for batch_idx in range(b):
            first_pose = first_ego_pose_world[batch_idx].float()
            world = (first_pose @ centers_h.t()).t()
            for time_idx in range(t):
                alpha_flat = alpha[batch_idx, time_idx].reshape(-1).float()
                sem_flat = sem_prob[batch_idx, time_idx].reshape(self.projected_classes, -1).float()
                for view_idx in range(camera_to_world.shape[2]):
                    out[batch_idx, time_idx, view_idx] = self._project_one_view(
                        world,
                        alpha_flat,
                        sem_flat,
                        camera_intrinsics[batch_idx, view_idx],
                        camera_to_world[batch_idx, time_idx, view_idx],
                        height=height,
                        width=width,
                    ).to(out.dtype)
        return out

    def _voxel_centers(self, *, nx: int, ny: int, nz: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        xs = torch.linspace(
            self.x_range[0] + 0.5 * self.voxel_size,
            self.x_range[1] - 0.5 * self.voxel_size,
            nx,
            device=device,
            dtype=dtype,
        )
        ys = torch.linspace(
            self.y_range[0] + 0.5 * self.voxel_size,
            self.y_range[1] - 0.5 * self.voxel_size,
            ny,
            device=device,
            dtype=dtype,
        )
        zs = torch.linspace(
            self.z_range[0] + 0.5 * self.voxel_size,
            self.z_range[1] - 0.5 * self.voxel_size,
            nz,
            device=device,
            dtype=dtype,
        )
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    def _project_one_view(
        self,
        world_xyz_h: torch.Tensor,
        alpha_flat: torch.Tensor,
        sem_flat: torch.Tensor,
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        world_xyz_h = world_xyz_h.float()
        alpha_flat = alpha_flat.float()
        sem_flat = sem_flat.float()
        intrinsics = intrinsics.float()
        cam_to_world = cam_to_world.float()
        fx, fy, cx, cy = intrinsics
        world_to_cam = torch.linalg.inv(cam_to_world)
        cam = (world_to_cam @ world_xyz_h.t()).t()[:, :3]
        z = cam[:, 2]
        valid = (z > 1e-3) & (alpha_flat > 1e-4) & torch.isfinite(cam).all(dim=-1)
        sem_num = torch.zeros((self.projected_classes, height * width), device=cam.device, dtype=torch.float32)
        alpha_den = torch.zeros((height * width,), device=cam.device, dtype=torch.float32)
        if not torch.any(valid):
            return sem_num.reshape(self.projected_classes, height, width)

        cam = cam[valid]
        z = z[valid]
        alpha_flat = alpha_flat[valid]
        sem_flat = sem_flat[:, valid]
        u = cam[:, 0] * fx / z + cx
        v = cam[:, 1] * fy / z + cy
        valid = (u >= -1.0) & (u <= width) & (v >= -1.0) & (v <= height)
        if not torch.any(valid):
            return sem_num.reshape(self.projected_classes, height, width)

        u = u[valid]
        v = v[valid]
        alpha_flat = alpha_flat[valid]
        sem_flat = sem_flat[:, valid]
        x0 = torch.floor(u).long()
        y0 = torch.floor(v).long()
        neighbors = ((x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1))
        for xn, yn in neighbors:
            keep = (xn >= 0) & (xn < width) & (yn >= 0) & (yn < height)
            if not torch.any(keep):
                continue
            xu = xn[keep]
            yu = yn[keep]
            du = (1.0 - (u[keep] - xu.float()).abs()).clamp(0.0, 1.0)
            dv = (1.0 - (v[keep] - yu.float()).abs()).clamp(0.0, 1.0)
            weight = du * dv * alpha_flat[keep]
            idx = yu * width + xu
            alpha_den.scatter_add_(0, idx, weight)
            sem_num.scatter_add_(1, idx.unsqueeze(0).expand(self.projected_classes, -1), sem_flat[:, keep] * weight.unsqueeze(0))
        return (sem_num / alpha_den.clamp_min(1e-6).unsqueeze(0)).reshape(self.projected_classes, height, width)
