"""DGGT-style Gaussian rasterization renderer backed by gsplat."""

from __future__ import annotations

import torch
import torch.nn as nn

from dvgt_occ.types import GaussianAssignmentOutput, GaussianOutput, RenderOutput

_GSPLAT_IMPORT_ERROR: Exception | None = None
try:
    from gsplat.rendering import rasterization
except ImportError as exc:  # pragma: no cover - runtime dependency in training env
    rasterization = None
    _GSPLAT_IMPORT_ERROR = exc


class GaussianMaskRenderer(nn.Module):
    def __init__(
        self,
        output_size: tuple[int, int] = (224, 448),
        splat_radius: int = 2,
        scale_max: float = 0.30,
        rasterize_mode: str = "classic",
        keep_score_gate: float = 0.0,
        keep_topk_ratio: float = 0.0,
        anti_grid_replicas: int = 1,
        anti_grid_jitter_m: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.splat_radius = max(int(splat_radius), 1)
        self.scale_max = float(scale_max)
        self.rasterize_mode = str(rasterize_mode)
        self.keep_score_gate = float(keep_score_gate)
        self.keep_topk_ratio = float(keep_topk_ratio)
        self.anti_grid_replicas = max(int(anti_grid_replicas), 1)
        self.anti_grid_jitter_m = max(float(anti_grid_jitter_m), 0.0)

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
        candidate_view_mask: torch.Tensor | None = None,
    ) -> RenderOutput:
        if rasterization is None:
            detail = f" ({_GSPLAT_IMPORT_ERROR!r})" if _GSPLAT_IMPORT_ERROR is not None else ""
            raise ImportError(f"gsplat.rendering.rasterization is required for Gaussian rendering{detail}.")

        b, t, v, h, w = gaussians.opacity.shape[:5]
        opacity = gaussians.opacity.squeeze(-1).clamp(0.0, 1.0)
        if self.keep_score_gate > 0.0:
            gate = max(0.0, min(self.keep_score_gate, 1.0))
            keep = gaussians.keep_score.squeeze(-1).clamp(0.0, 1.0)
            base_alpha = opacity * ((1.0 - gate) + gate * keep)
        else:
            base_alpha = opacity

        dynamic_prob = gaussians.dynamic_prob
        if dynamic_prob is not None:
            dynamicness = dynamic_prob.squeeze(-1).clamp(0.0, 1.0)
        else:
            background = assignment.background_prob.reshape(b, t, v, h, w)
            dynamicness = (1.0 - background).clamp(0.0, 1.0)
        alpha_all = base_alpha
        alpha_dyn = base_alpha * dynamicness
        alpha_sta = base_alpha * (1.0 - dynamicness)

        rgb_all, render_alpha_all, sigma_mean, touch_ratio = self._render_branch(
            gaussians.center,
            gaussians.scale,
            gaussians.rotation,
            gaussians.feat_dc,
            alpha_all,
            camera_intrinsics,
            camera_to_world,
            first_ego_pose_world,
            candidate_view_mask=candidate_view_mask,
            return_rgb=True,
        )
        rgb_dyn, render_alpha_dyn, _, _ = self._render_branch(
            gaussians.center,
            gaussians.scale,
            gaussians.rotation,
            gaussians.feat_dc,
            alpha_dyn,
            camera_intrinsics,
            camera_to_world,
            first_ego_pose_world,
            candidate_view_mask=candidate_view_mask,
            return_rgb=compute_dynamic_rgb,
        )
        if compute_static_branch:
            rgb_sta, render_alpha_sta, _, _ = self._render_branch(
                gaussians.center,
                gaussians.scale,
                gaussians.rotation,
                gaussians.feat_dc,
                alpha_sta,
                camera_intrinsics,
                camera_to_world,
                first_ego_pose_world,
                candidate_view_mask=candidate_view_mask,
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
        rotations: torch.Tensor,
        colors: torch.Tensor,
        alpha: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        first_ego_pose_world: torch.Tensor,
        *,
        candidate_view_mask: torch.Tensor | None,
        return_rgb: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, v, h, w, _ = centers.shape
        height, width = self.output_size
        device = centers.device
        dtype = centers.dtype
        num_views = camera_to_world.shape[2]

        out_rgb = torch.zeros((b, t, num_views, 3, height, width), device=device, dtype=dtype)
        out_alpha = torch.zeros((b, t, num_views, 1, height, width), device=device, dtype=dtype)
        sigma_total = torch.zeros((), device=device, dtype=torch.float32)
        touch_total = torch.zeros((), device=device, dtype=torch.float32)
        view_count = 0

        first_rot_quat = self._matrix_to_quaternion(first_ego_pose_world[:, :3, :3].float())
        n_total = v * h * w

        for batch_idx in range(b):
            first_pose = first_ego_pose_world[batch_idx]
            for time_idx in range(t):
                xyz = centers[batch_idx, time_idx].reshape(-1, 3)
                ones_v = torch.ones((n_total, 1), device=device, dtype=dtype)
                xyz_h = torch.cat([xyz, ones_v], dim=-1)
                means_world = (first_pose @ xyz_h.t()).t()[:, :3].float()
                quats_world = self._quat_mul(
                    first_rot_quat[batch_idx].unsqueeze(0).expand(n_total, -1),
                    rotations[batch_idx, time_idx].reshape(-1, 4).float(),
                )
                scales_v = scales[batch_idx, time_idx].reshape(-1, 3).clamp_min(1e-4).float()
                if self.scale_max > 0.0:
                    scales_v = scales_v.clamp(max=self.scale_max)
                colors_v = colors[batch_idx, time_idx].reshape(-1, 3).clamp(0.0, 1.0).float()
                for view_idx in range(num_views):
                    opacities = alpha[batch_idx, time_idx].reshape(-1).clamp(0.0, 1.0).float()
                    if candidate_view_mask is not None:
                        view_keep = candidate_view_mask[batch_idx, view_idx].to(device=device, dtype=opacities.dtype)
                        if view_keep.numel() == v:
                            view_keep = view_keep.reshape(v, 1, 1).expand(v, h, w).reshape(-1)
                            opacities = opacities * view_keep

                    viewmat = torch.linalg.inv(camera_to_world[batch_idx, time_idx, view_idx:view_idx+1].float())
                    k = self._build_k_mats(camera_intrinsics[batch_idx, view_idx:view_idx+1])
                    rgb_v, alpha_v, sigma_v, touch_v = self._rasterize_views(
                        means_world=means_world,
                        quats_world=quats_world,
                        scales=scales_v,
                        opacities=opacities,
                        colors=colors_v,
                        viewmats=viewmat,
                        ks=k,
                        width=width,
                        height=height,
                        return_rgb=return_rgb,
                    )
                    out_rgb[batch_idx, time_idx, view_idx] = rgb_v[0].to(dtype)
                    out_alpha[batch_idx, time_idx, view_idx] = alpha_v[0].to(dtype)
                    sigma_total = sigma_total + sigma_v.float()
                    touch_total = touch_total + touch_v.float()
                    view_count += 1

        denom = max(view_count, 1)
        return out_rgb, out_alpha, (sigma_total / denom).to(dtype), (touch_total / denom).to(dtype)

    def _rasterize_views(
        self,
        *,
        means_world: torch.Tensor,
        quats_world: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        viewmats: torch.Tensor,
        ks: torch.Tensor,
        width: int,
        height: int,
        return_rgb: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = means_world.device
        valid = (
            torch.isfinite(means_world).all(dim=-1)
            & torch.isfinite(quats_world).all(dim=-1)
            & torch.isfinite(scales).all(dim=-1)
            & torch.isfinite(colors).all(dim=-1)
            & torch.isfinite(opacities)
            & (opacities > 1e-6)
        )
        means_h = torch.cat(
            [means_world.float(), torch.ones((means_world.shape[0], 1), device=device, dtype=torch.float32)],
            dim=-1,
        )
        camera_z = (viewmats[0].float() @ means_h.t()).t()[:, 2]
        valid = valid & torch.isfinite(camera_z) & (camera_z.abs() > 0.5)
        if 0.0 < self.keep_topk_ratio < 1.0 and torch.any(valid):
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            keep_n = max(1, int(round(float(valid_idx.numel()) * self.keep_topk_ratio)))
            if keep_n < int(valid_idx.numel()):
                score = opacities[valid_idx].detach()
                # Break the uniform-initialization tie without introducing
                # stochastic render noise; otherwise top-k picks a rectangular
                # prefix of the 56x112 lattice and preserves the grid artifact.
                idx_f = valid_idx.to(dtype=torch.float32)
                tie_break = torch.frac(torch.sin((idx_f + 1.0) * 12.9898) * 43758.5453) * 1.0e-4
                _, local_keep = torch.topk(score + tie_break.to(score.device), k=keep_n, largest=True, sorted=False)
                pruned_valid = torch.zeros_like(valid)
                pruned_valid[valid_idx[local_keep]] = True
                valid = pruned_valid
        if not torch.any(valid):
            num_views = viewmats.shape[0]
            zero_rgb = torch.zeros((num_views, 3, height, width), device=device, dtype=means_world.dtype)
            zero_alpha = torch.zeros((num_views, 1, height, width), device=device, dtype=means_world.dtype)
            zero = torch.zeros((), device=device, dtype=means_world.dtype)
            return zero_rgb, zero_alpha, zero, zero

        means = means_world[valid]
        quats = torch.nn.functional.normalize(quats_world[valid], dim=-1)
        scl = scales[valid]
        opa = opacities[valid]
        cols = colors[valid] if return_rgb else torch.zeros_like(colors[valid])
        means, quats, scl, opa, cols = self._expand_antigrid_replicas(
            means=means,
            quats=quats,
            scales=scl,
            opacities=opa,
            colors=cols,
            viewmats=viewmats,
        )

        raster_kwargs = {
            "means": means,
            "quats": quats,
            "scales": scl,
            "opacities": opa,
            "colors": cols,
            "viewmats": viewmats,
            "Ks": ks,
            "width": width,
            "height": height,
        }
        if self.rasterize_mode and self.rasterize_mode != "classic":
            raster_kwargs["rasterize_mode"] = self.rasterize_mode
        try:
            renders, alphas, aux = rasterization(**raster_kwargs)
        except TypeError as exc:
            if "rasterize_mode" not in raster_kwargs:
                raise
            raster_kwargs.pop("rasterize_mode")
            renders, alphas, aux = rasterization(**raster_kwargs)
        renders = renders.permute(0, 3, 1, 2).contiguous()
        alphas = alphas.permute(0, 3, 1, 2).contiguous()
        if not return_rgb:
            renders.zero_()

        sigma_mean = scl.mean()
        if isinstance(aux, dict) and "radii" in aux:
            radii = aux["radii"]
            if torch.is_tensor(radii) and radii.numel() > 0:
                sigma_mean = radii.float().mean()
        touch_ratio = (alphas > 1e-6).float().mean()
        return renders, alphas, sigma_mean, touch_ratio

    def _expand_antigrid_replicas(
        self,
        *,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        viewmats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        replicas = self.anti_grid_replicas
        jitter_m = self.anti_grid_jitter_m
        if replicas <= 1 or jitter_m <= 0.0 or means.numel() == 0:
            return means, quats, scales, opacities, colors

        n = means.shape[0]
        device = means.device
        dtype = means.dtype
        cam_to_world = torch.linalg.inv(viewmats[0].float())
        right = torch.nn.functional.normalize(cam_to_world[:3, 0], dim=0).to(device=device, dtype=dtype)
        up = torch.nn.functional.normalize(cam_to_world[:3, 1], dim=0).to(device=device, dtype=dtype)

        idx = torch.arange(n, device=device, dtype=torch.float32)
        offsets = []
        for replica_idx in range(replicas):
            if replica_idx == 0:
                jx = torch.zeros_like(idx)
                jy = torch.zeros_like(idx)
            else:
                salt_x = 12.9898 + 19.19 * float(replica_idx)
                salt_y = 78.2330 + 37.37 * float(replica_idx)
                jx = torch.frac(torch.sin((idx + 1.0) * salt_x) * 43758.5453) * 2.0 - 1.0
                jy = torch.frac(torch.sin((idx + 1.0) * salt_y) * 24634.6345) * 2.0 - 1.0
            offsets.append((jx[:, None].to(dtype) * right + jy[:, None].to(dtype) * up) * jitter_m)
        jitter = torch.stack(offsets, dim=1)

        means_rep = (means[:, None, :] + jitter).reshape(n * replicas, 3)
        quats_rep = quats[:, None, :].expand(n, replicas, 4).reshape(n * replicas, 4)
        scales_rep = scales[:, None, :].expand(n, replicas, 3).reshape(n * replicas, 3)
        opacities_rep = (opacities[:, None].expand(n, replicas) / float(replicas)).reshape(n * replicas)
        colors_rep = colors[:, None, :].expand(n, replicas, colors.shape[-1]).reshape(n * replicas, colors.shape[-1])
        return means_rep, quats_rep, scales_rep, opacities_rep, colors_rep

    @staticmethod
    def _build_k_mats(intrinsics: torch.Tensor) -> torch.Tensor:
        intrinsics = intrinsics.float()
        fx = intrinsics[:, 0]
        fy = intrinsics[:, 1]
        cx = intrinsics[:, 2]
        cy = intrinsics[:, 3]
        ks = torch.zeros((intrinsics.shape[0], 3, 3), device=intrinsics.device, dtype=intrinsics.dtype)
        ks[:, 0, 0] = fx
        ks[:, 1, 1] = fy
        ks[:, 0, 2] = cx
        ks[:, 1, 2] = cy
        ks[:, 2, 2] = 1.0
        return ks

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )

    @staticmethod
    def _matrix_to_quaternion(rot: torch.Tensor) -> torch.Tensor:
        m = rot
        trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
        qw = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 0.5
        qx = (m[..., 2, 1] - m[..., 1, 2]) / (4.0 * qw.clamp_min(1e-8))
        qy = (m[..., 0, 2] - m[..., 2, 0]) / (4.0 * qw.clamp_min(1e-8))
        qz = (m[..., 1, 0] - m[..., 0, 1]) / (4.0 * qw.clamp_min(1e-8))
        quat = torch.stack((qw, qx, qy, qz), dim=-1)
        return torch.nn.functional.normalize(quat, dim=-1)
