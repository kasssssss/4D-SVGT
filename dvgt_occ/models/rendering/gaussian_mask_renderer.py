"""Camera-aware differentiable Gaussian image renderer."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from dvgt_occ.types import GaussianAssignmentOutput, GaussianOutput, RenderOutput


class GaussianMaskRenderer(nn.Module):
    def __init__(self, output_size: tuple[int, int] = (224, 448), splat_radius: int = 2) -> None:
        super().__init__()
        self.output_size = output_size
        self.splat_radius = max(int(splat_radius), 1)
        self.radius_scale = 1.5
        offsets = list(itertools.product(range(-self.splat_radius, self.splat_radius + 1), repeat=2))
        self.register_buffer('kernel_offsets', torch.tensor(offsets, dtype=torch.long), persistent=False)

    def forward(
        self,
        gaussians: GaussianOutput,
        assignment: GaussianAssignmentOutput,
        sem_proj_2d: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        first_ego_pose_world: torch.Tensor,
        *,
        compute_static_branch: bool = True,
        compute_dynamic_rgb: bool = True,
    ) -> RenderOutput:
        b, t, v, h, w = gaussians.opacity.shape[:5]
        opacity = gaussians.opacity.squeeze(-1)
        base_alpha = opacity.clamp(0.0, 1.0)

        background = assignment.background_prob.reshape(b, t, v, h, w)
        dynamicness = (1.0 - background).clamp(0.0, 1.0)
        alpha_all = base_alpha
        alpha_dyn = base_alpha * dynamicness
        alpha_sta = base_alpha * (1.0 - dynamicness)

        rgb_all, render_alpha_all, sigma_mean, touch_ratio = self._render_branch(
            gaussians.center,
            gaussians.scale,
            gaussians.feat_dc,
            alpha_all,
            camera_intrinsics,
            camera_to_world,
            first_ego_pose_world,
            return_rgb=True,
        )
        rgb_dyn, render_alpha_dyn, _, _ = self._render_branch(
            gaussians.center,
            gaussians.scale,
            gaussians.feat_dc,
            alpha_dyn,
            camera_intrinsics,
            camera_to_world,
            first_ego_pose_world,
            return_rgb=compute_dynamic_rgb,
        )
        if compute_static_branch:
            rgb_sta, render_alpha_sta, _, _ = self._render_branch(
                gaussians.center,
                gaussians.scale,
                gaussians.feat_dc,
                alpha_sta,
                camera_intrinsics,
                camera_to_world,
                first_ego_pose_world,
                return_rgb=True,
            )
        else:
            rgb_sta = torch.zeros_like(rgb_all)
            render_alpha_sta = torch.zeros_like(render_alpha_all)
        return RenderOutput(
            render_rgb_static=rgb_sta,
            render_rgb_dynamic=rgb_dyn,
            render_rgb_all=rgb_all,
            render_alpha_static=render_alpha_sta,
            render_alpha_dynamic=render_alpha_dyn,
            render_alpha_all=render_alpha_all,
            sem_proj_2d=sem_proj_2d,
            debug_mean_sigma=sigma_mean,
            debug_touch_ratio=touch_ratio,
        )

    def _render_branch(
        self,
        centers: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        alpha: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        first_ego_pose_world: torch.Tensor,
        *,
        return_rgb: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _, _, _, _ = centers.shape
        height, width = self.output_size
        device = centers.device
        rgb = torch.zeros((b, t, camera_to_world.shape[2], 3, height, width), device=device, dtype=centers.dtype) if return_rgb else None
        out_alpha = torch.zeros((b, t, camera_to_world.shape[2], 1, height, width), device=device, dtype=centers.dtype)
        sigma_total = torch.zeros((), device=device, dtype=centers.dtype)
        touch_total = torch.zeros((), device=device, dtype=centers.dtype)
        view_count = torch.zeros((), device=device, dtype=centers.dtype)

        centers_flat = centers.reshape(b, t, -1, 3)
        scales_flat = scales.mean(dim=-1).reshape(b, t, -1)
        colors_flat = colors.reshape(b, t, -1, 3).clamp(0.0, 1.0)
        alpha_flat = alpha.reshape(b, t, -1).clamp(0.0, 1.0)

        ones = torch.ones((centers_flat.shape[2], 1), device=device, dtype=centers.dtype)
        for batch_idx in range(b):
            first_pose = first_ego_pose_world[batch_idx]
            for time_idx in range(t):
                xyz = centers_flat[batch_idx, time_idx]
                xyz_h = torch.cat([xyz, ones], dim=-1)
                world = (first_pose @ xyz_h.t()).t()
                world_to_cam_all = torch.linalg.inv(camera_to_world[batch_idx, time_idx].float())
                for view_idx in range(camera_to_world.shape[2]):
                    rgb_view, alpha_view, sigma_view, touch_view = self._project_and_splat(
                        world,
                        scales_flat[batch_idx, time_idx],
                        colors_flat[batch_idx, time_idx],
                        alpha_flat[batch_idx, time_idx],
                        camera_intrinsics[batch_idx, view_idx],
                        world_to_cam_all[view_idx],
                        height=height,
                        width=width,
                        return_rgb=return_rgb,
                    )
                    if rgb is not None:
                        rgb[batch_idx, time_idx, view_idx] = rgb_view
                    out_alpha[batch_idx, time_idx, view_idx, 0] = alpha_view
                    sigma_total = sigma_total + sigma_view
                    touch_total = touch_total + touch_view
                    view_count = view_count + 1.0
        denom = view_count.clamp_min(1.0)
        if rgb is None:
            rgb = torch.zeros((b, t, camera_to_world.shape[2], 3, height, width), device=device, dtype=centers.dtype)
        return rgb, out_alpha, sigma_total / denom, touch_total / denom

    def _project_and_splat(
        self,
        world_xyz_h: torch.Tensor,
        scale: torch.Tensor,
        color: torch.Tensor,
        alpha: torch.Tensor,
        intrinsics: torch.Tensor,
        world_to_cam: torch.Tensor,
        *,
        height: int,
        width: int,
        return_rgb: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = world_xyz_h.device
        out_dtype = world_xyz_h.dtype
        world_xyz_h = world_xyz_h.float()
        scale = scale.float()
        color = color.float()
        alpha = alpha.float()
        intrinsics = intrinsics.float()
        world_to_cam = world_to_cam.float()
        fx, fy, cx, cy = intrinsics
        cam = (world_to_cam @ world_xyz_h.t()).t()[:, :3]
        z = cam[:, 2]
        valid = (z > 1e-3) & torch.isfinite(cam).all(dim=-1) & (alpha > 1e-5)
        rgb_num = torch.zeros((3, height * width), device=device, dtype=torch.float32) if return_rgb else None
        alpha_den = torch.zeros((height * width,), device=device, dtype=torch.float32)
        if not torch.any(valid):
            zero = torch.zeros((), device=device, dtype=out_dtype)
            rgb_out = torch.zeros((3, height, width), device=device, dtype=out_dtype)
            return rgb_out, alpha_den.reshape(height, width).to(out_dtype), zero, zero

        cam = cam[valid]
        scale = scale[valid]
        color = color[valid]
        alpha = alpha[valid]
        z = z[valid]

        u = cam[:, 0] * fx / z + cx
        v = cam[:, 1] * fy / z + cy
        sigma = (((fx + fy) * 0.5) * scale.abs() / z.clamp_min(1e-3)).clamp(0.75, 10.0)
        valid = (u >= -self.splat_radius - 1.0) & (u <= width + self.splat_radius) & (v >= -self.splat_radius - 1.0) & (v <= height + self.splat_radius)
        if not torch.any(valid):
            zero = torch.zeros((), device=device, dtype=out_dtype)
            rgb_out = torch.zeros((3, height, width), device=device, dtype=out_dtype)
            return rgb_out, alpha_den.reshape(height, width).to(out_dtype), zero, zero

        u = u[valid]
        v = v[valid]
        sigma = sigma[valid]
        color = color[valid]
        alpha = alpha[valid]
        z = z[valid]
        sigma_mean = sigma.mean().to(out_dtype)

        order = torch.argsort(z, descending=False)
        u = u[order]
        v = v[order]
        sigma = sigma[order]
        color = color[order]
        alpha = alpha[order]

        x0 = torch.floor(u).long()
        y0 = torch.floor(v).long()
        offsets = self.kernel_offsets.to(device=device)
        radii = torch.ceil(sigma * self.radius_scale).long().clamp(min=1, max=self.splat_radius)
        xo_all = x0[:, None] + offsets[None, :, 0]
        yo_all = y0[:, None] + offsets[None, :, 1]
        radius_keep = (offsets[None, :, 0].abs() <= radii[:, None]) & (offsets[None, :, 1].abs() <= radii[:, None])
        bounds_keep = (xo_all >= 0) & (xo_all < width) & (yo_all >= 0) & (yo_all < height)
        keep_all = radius_keep & bounds_keep

        sigma_safe = sigma.clamp_min(1e-3)
        du_all = (u[:, None] - xo_all.float()) / sigma_safe[:, None]
        dv_all = (v[:, None] - yo_all.float()) / sigma_safe[:, None]
        local_alpha_all = torch.exp(-0.5 * (du_all * du_all + dv_all * dv_all)) * alpha[:, None]
        local_alpha_all = local_alpha_all.clamp(0.0, 0.999)
        idx_all = yo_all * width + xo_all

        for point_idx in range(u.shape[0]):
            keep = keep_all[point_idx]
            if not torch.any(keep):
                continue
            idx = idx_all[point_idx, keep]
            local_alpha = local_alpha_all[point_idx, keep]
            transmittance = (1.0 - alpha_den[idx]).clamp(0.0, 1.0)
            contrib = local_alpha * transmittance
            alpha_den[idx] = (alpha_den[idx] + contrib).clamp(0.0, 0.999)
            if rgb_num is not None:
                rgb_num[:, idx] = rgb_num[:, idx] + color[point_idx].unsqueeze(-1) * contrib.unsqueeze(0)

        if rgb_num is None:
            rgb = torch.zeros((3, height, width), device=device, dtype=out_dtype)
        else:
            rgb = rgb_num.reshape(3, height, width).clamp(0.0, 1.0).to(out_dtype)
        touch_ratio = (alpha_den > 1e-6).float().mean().to(out_dtype)
        return (
            rgb,
            alpha_den.reshape(height, width).clamp(0.0, 1.0).to(out_dtype),
            sigma_mean,
            touch_ratio,
        )
