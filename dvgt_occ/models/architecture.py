"""Top-level scaffold that wires the first model interfaces together."""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dvgt_occ.config import DEFAULT_DVGT_OCC_CONFIG, DVGTOccConfig
from dvgt_occ.models.bridges import GSOccGlobalLatentBridge
from dvgt_occ.models.dynamic import DynamicDenseBranch, DynamicQueryDecoder
from dvgt_occ.models.gaussian import (
    EntityAggregator,
    GSOccLocalBridge,
    GaussianMultiViewMerge,
    GaussianQueryAssignment,
    GSHead,
)
from dvgt_occ.models.occupancy import OccHead
from dvgt_occ.models.projection import OccSemanticProjector
from dvgt_occ.models.reassembly import TokenReassembly
from dvgt_occ.models.rendering import GaussianMaskRenderer
from dvgt_occ.models.sky import SkyRayBackground
from dvgt_occ.types import BridgeOutput, DynamicDenseOutput, EntityOutput, GaussianAssignmentOutput, SkyOutput


class DVGTOccModel(nn.Module):
    def __init__(self, config: DVGTOccConfig = DEFAULT_DVGT_OCC_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.reassembly = TokenReassembly(
            selected_layers=config.selected_layers,
            in_dim=config.joint_token_dim,
            out_dim=config.neck_dim,
            patch_grid=config.patch_grid,
        )
        self.gs_reassembly = TokenReassembly(
            selected_layers=config.selected_layers,
            in_dim=config.joint_token_dim,
            out_dim=config.neck_dim,
            patch_grid=config.patch_grid,
        )
        full_size = (config.image_height, config.image_width)
        quarter_size = (max(1, config.image_height // 4), max(1, config.image_width // 4))
        eighth_size = (max(1, config.image_height // 8), max(1, config.image_width // 8))
        self.dynamic_dense = DynamicDenseBranch(channels=config.neck_dim, full_channels=config.full_dim, output_size=full_size)
        self.occ_head = OccHead(
            channels=config.neck_dim,
            semantic_classes=config.semantic_classes,
            dynamic_channels=config.dynamic_dense_dim,
            lift_channels=config.occ_lift_dim,
            bottleneck_channels=config.occ_bottleneck_dim,
            occ_shape_zyx=config.occ_grid.shape_zyx,
            x_range=config.occ_grid.x_range,
            y_range=config.occ_grid.y_range,
            z_range=config.occ_grid.z_range,
            voxel_size=config.occ_grid.voxel_size,
            output_size=full_size,
        )
        self.dynamic_query = DynamicQueryDecoder(
            query_dim=config.dynamic_query_dim,
            max_track_queries=config.max_track_queries,
            new_queries=config.new_queries,
            dynamic_classes=config.dynamic_classes,
            motion_dim=7,
            local_feature_dim=config.dynamic_dense_dim,
            local_topk=min(config.sparse_dynamic_anchors, config.num_views * quarter_size[0] * quarter_size[1]),
            occ_samples_per_scale=config.occ_samples_per_scale,
            x_range=config.occ_grid.x_range,
            y_range=config.occ_grid.y_range,
            z_range=config.occ_grid.z_range,
        )
        self.gs_head = GSHead(
            channels=config.neck_dim,
            feature_dim=config.gs_feature_dim,
            instance_dim=config.instance_dim,
            motion_dim=config.motion_dim,
            bias_scale=config.gs_bias_scale,
            init_scale=config.gs_init_scale,
            init_opacity_logit=config.gs_init_opacity_logit,
            anchor_jitter_m=config.gs_anchor_jitter_m,
            output_level=config.gs_output_level,
            scale_multiplier=config.gs_scale_multiplier,
            output_size=full_size,
        )
        self.gaussian_merge = GaussianMultiViewMerge()
        self.gaussian_assignment = GaussianQueryAssignment(
            query_dim=config.dynamic_query_dim,
            gaussian_feat_dim=config.instance_dim + config.motion_dim,
        )
        self.gs_occ_local_bridge = GSOccLocalBridge(
            occ_channels=config.occ_base_dim,
            gs_channels=config.instance_dim + config.motion_dim,
            grid_min_xyz=(config.occ_grid.x_range[0], config.occ_grid.y_range[0], config.occ_grid.z_range[0]),
            voxel_size=config.occ_grid.voxel_size,
            grid_shape_zyx=config.occ_grid.shape_zyx,
        )
        self.gs_occ_global_latent_bridge = GSOccGlobalLatentBridge(channels=config.neck_dim, num_latents=config.global_latents)
        self.gs_token_proj = nn.Linear(config.instance_dim + config.motion_dim, config.neck_dim)
        self.occ_latent_proj = nn.Linear(config.occ_bottleneck_dim, config.neck_dim)
        self.global_bridge_to_entity = nn.Linear(config.neck_dim, config.instance_dim + config.motion_dim)
        self.entity_aggregator = EntityAggregator(
            instance_dim=config.instance_dim,
            motion_dim=config.motion_dim,
            query_dim=config.dynamic_query_dim,
        )
        self.occ_semantic_projector = OccSemanticProjector(
            semantic_classes=config.semantic_classes,
            projected_classes=config.projected_semantic_classes,
            output_size=(config.image_height, config.image_width),
            x_range=config.occ_grid.x_range,
            y_range=config.occ_grid.y_range,
            z_range=config.occ_grid.z_range,
            voxel_size=config.occ_grid.voxel_size,
        )
        self.mask_renderer = GaussianMaskRenderer(
            output_size=(config.image_height, config.image_width),
            splat_radius=config.render_splat_radius,
            scale_max=config.render_scale_max,
            rasterize_mode=config.render_rasterize_mode,
            keep_score_gate=config.render_keep_score_gate,
            keep_topk_ratio=config.render_keep_topk_ratio,
            anti_grid_replicas=config.render_antigrid_replicas,
            anti_grid_jitter_m=config.render_antigrid_jitter_m,
        )
        self.sky_model = SkyRayBackground(
            hidden_dim=config.sky_hidden_dim,
            fourier_freqs=config.sky_fourier_freqs,
        )
        self.train_target = "joint"

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.dynamic_dense.set_gradient_checkpointing(enabled)
        self.occ_head.set_gradient_checkpointing(enabled)
        self.gs_head.set_gradient_checkpointing(enabled)

    def set_train_target(self, train_target: str) -> None:
        self.train_target = str(train_target)

    def configure_optional_modules(self, active_loss_weights: dict | None) -> None:
        return

    def forward(self, batch: dict) -> dict:
        active_loss_weights = batch.get("_active_loss_weights")
        profile_timing = bool(batch.get("_profile_timing", False))
        timings_ms: dict[str, float] = {}
        train_mode = str(batch.get("_train_mode", "v1-stable"))

        def _record_timing(name: str, fn):
            if not profile_timing:
                return fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings_ms[name] = (time.perf_counter() - start) * 1000.0
            return out

        def _weight(name: str) -> float:
            if not isinstance(active_loss_weights, dict):
                return 1.0
            return float(active_loss_weights.get(name, 0.0))

        optimize_for_training = isinstance(active_loss_weights, dict)
        render_full_branches = not optimize_for_training
        use_gs_aux = self.train_target == "gs_only" and train_mode == "v1-gs-aux"
        need_features = self.train_target in ("occ_only", "joint") or use_gs_aux
        need_gs_features = self.train_target in ("gs_only", "joint")

        points = batch["points"]
        points_conf = batch["points_conf"]
        quarter_size = (max(1, self.config.image_height // 4), max(1, self.config.image_width // 4))
        eighth_size = (max(1, self.config.image_height // 8), max(1, self.config.image_width // 8))
        xyz_1_4 = self._downsample_points(points, size=quarter_size)
        conf_1_4 = self._downsample_conf(points_conf, size=quarter_size)
        xyz_1_8 = self._downsample_points(points, size=eighth_size)
        conf_1_8 = self._downsample_conf(points_conf, size=eighth_size)
        rgb_for_gs = batch.get("source_rgb", batch.get("rgb_source", batch.get("rgb_target")))
        use_full_res_gs = str(getattr(self.config, "gs_output_level", "p2")).lower() == "full"
        if use_full_res_gs:
            xyz_gs = points.permute(0, 1, 2, 5, 3, 4).contiguous()
            conf_gs = points_conf[:, :, :, None].contiguous()
            rgb_gs = rgb_for_gs
        else:
            xyz_gs = xyz_1_4
            conf_gs = conf_1_4
            rgb_gs = None
        if rgb_for_gs is not None:
            b_r, t_r, v_r, c_r, h_r, w_r = rgb_for_gs.shape
            if rgb_gs is None:
                rgb_gs = F.interpolate(
                    rgb_for_gs.reshape(b_r * t_r * v_r, c_r, h_r, w_r),
                    size=quarter_size, mode="bilinear", align_corners=False,
                ).reshape(b_r, t_r, v_r, c_r, quarter_size[0], quarter_size[1])
        else:
            rgb_gs = None
        features = None
        if need_features:
            with torch.autograd.profiler.record_function("dvgt_occ/reassembly"):
                features = _record_timing(
                    "reassembly",
                    lambda: self.reassembly(batch["aggregated_tokens"], batch["raw_patch_tokens"]),
                )
        gs_features = None
        if need_gs_features:
            with torch.autograd.profiler.record_function("dvgt_occ/gs_reassembly"):
                gs_features = _record_timing(
                    "gs_reassembly",
                    lambda: self.gs_reassembly(batch["aggregated_tokens"], batch["raw_patch_tokens"]),
                )
        if need_features:
            with torch.autograd.profiler.record_function("dvgt_occ/dynamic_dense"):
                dynamic = _record_timing(
                    "dynamic_dense",
                    lambda: self.dynamic_dense(features, xyz_1_4=xyz_1_4, conf_1_4=conf_1_4),
                )
        else:
            dynamic = self._build_dummy_dynamic(xyz_1_4=xyz_1_4, dtype=xyz_1_4.dtype)
        if self.train_target == "occ_only":
            with torch.autograd.profiler.record_function("dvgt_occ/occ_head"):
                occ = _record_timing("occ_head", lambda: self.occ_head(features, dynamic=dynamic, xyz_1_4=xyz_1_4, conf_1_4=conf_1_4))
            outputs = {
                "dynamic": dynamic,
                "occ": occ,
            }
            sky = self._run_sky_model(batch)
            if sky is not None:
                outputs["sky"] = sky
            if profile_timing:
                outputs["timings_ms"] = timings_ms
            return outputs

        if self.train_target == "gs_only":
            with torch.autograd.profiler.record_function("dvgt_occ/gs_head"):
                gaussians = _record_timing(
                    "gs_head",
                    lambda: self.gs_head(
                        gs_features,
                        dynamic=dynamic if use_gs_aux else None,
                        xyz_1_4=xyz_gs,
                        conf_1_4=conf_gs,
                        rgb_1_4=rgb_gs,
                    ),
                )
            with torch.autograd.profiler.record_function("dvgt_occ/gaussian_merge"):
                gaussians_merged = _record_timing("gaussian_merge", lambda: self.gaussian_merge(gaussians, global_track_id=None))
            dummy_assignment = self._build_dummy_assignment(gaussians_merged)
            dummy_sem_proj_2d = self._build_dummy_sem_proj(batch, gaussians_merged.center.dtype)
            candidate_view_mask = self._build_source_view_candidate_mask(batch, gaussians_merged) if self.config.source_render_own_view_only else None
            with torch.autograd.profiler.record_function("dvgt_occ/mask_renderer"):
                render = _record_timing(
                    "mask_renderer",
                    lambda: self.mask_renderer(
                        gaussians_merged,
                        dummy_assignment,
                        dummy_sem_proj_2d,
                        camera_intrinsics=batch["camera_intrinsics"],
                        camera_to_world=batch["camera_to_world"],
                        first_ego_pose_world=batch["first_ego_pose_world"],
                        compute_static_branch=False,
                        compute_dynamic_rgb=False,
                        candidate_view_mask=candidate_view_mask,
                    ),
                )
            outputs = {
                "dynamic": dynamic,
                "gaussians": gaussians_merged,
                "gaussians_pre_merge": gaussians,
                "assignment": dummy_assignment,
                "render": render,
            }
            sky = self._run_sky_model(batch)
            if sky is not None:
                outputs["sky"] = sky
                render.render_rgb_background = sky.render_rgb_background
                render.render_rgb_all_composited = (
                    render.render_alpha_all * render.render_rgb_all
                    + (1.0 - render.render_alpha_all) * sky.render_rgb_background
                )
            if profile_timing:
                outputs["timings_ms"] = timings_ms
            return outputs

        with torch.autograd.profiler.record_function("dvgt_occ/occ_head"):
            occ = _record_timing("occ_head", lambda: self.occ_head(features, dynamic=dynamic, xyz_1_4=xyz_1_4, conf_1_4=conf_1_4))
        with torch.autograd.profiler.record_function("dvgt_occ/dynamic_query"):
            queries = _record_timing(
                "dynamic_query",
                lambda: self.dynamic_query(
                    occ.occ_volumes,
                    dynamic_feat_1_4=dynamic.dyn_feat_1_4,
                    dynamic_logit_1_4=dynamic.dyn_logit_1_4,
                    dynamic_xyz_1_4=xyz_1_4,
                    batch=batch,
                ),
            )
        with torch.autograd.profiler.record_function("dvgt_occ/gs_head"):
            gaussians = _record_timing("gs_head", lambda: self.gs_head(gs_features, dynamic=dynamic, xyz_1_4=xyz_gs, conf_1_4=conf_gs, rgb_1_4=rgb_gs))
        global_track_id = batch.get("gs_global_track_id_1_4")
        with torch.autograd.profiler.record_function("dvgt_occ/gaussian_merge"):
            gaussians_merged = _record_timing("gaussian_merge", lambda: self.gaussian_merge(gaussians, global_track_id=global_track_id))
        with torch.autograd.profiler.record_function("dvgt_occ/query_assignment"):
            assignment = _record_timing("query_assignment", lambda: self.gaussian_assignment(gaussians_merged, queries))
        with torch.autograd.profiler.record_function("dvgt_occ/bridges"):
            bridges = _record_timing("bridges", lambda: self._run_bridges(gaussians_merged, occ))
        with torch.autograd.profiler.record_function("dvgt_occ/entity_aggregator"):
            entities = _record_timing("entity_aggregator", lambda: self.entity_aggregator(gaussians_merged, queries, assignment))
            entities = _record_timing("inject_bridge_context", lambda: self._inject_bridge_context(entities, bridges))
        with torch.autograd.profiler.record_function("dvgt_occ/occ_semantic_projector"):
            sem_proj_2d = _record_timing(
                "occ_semantic_projector",
                lambda: self.occ_semantic_projector(
                    occ.occ_logit,
                    occ.sem_logit,
                    camera_intrinsics=batch["camera_intrinsics"],
                    camera_to_world=batch["camera_to_world"],
                    first_ego_pose_world=batch["first_ego_pose_world"],
                ),
            )
        with torch.autograd.profiler.record_function("dvgt_occ/mask_renderer"):
            candidate_view_mask = self._build_source_view_candidate_mask(batch, gaussians_merged) if self.config.source_render_own_view_only else None
            render = _record_timing(
                "mask_renderer",
                lambda: self.mask_renderer(
                    gaussians_merged,
                    assignment,
                    sem_proj_2d,
                    camera_intrinsics=batch["camera_intrinsics"],
                    camera_to_world=batch["camera_to_world"],
                    first_ego_pose_world=batch["first_ego_pose_world"],
                    compute_static_branch=render_full_branches,
                    compute_dynamic_rgb=render_full_branches,
                    candidate_view_mask=candidate_view_mask,
                ),
            )
        outputs = {
            "dynamic": dynamic,
            "occ": occ,
            "queries": queries,
            "gaussians": gaussians_merged,
            "gaussians_pre_merge": gaussians,
            "assignment": assignment,
            "entities": entities,
            "bridges": bridges,
            "render": render,
        }
        sky = self._run_sky_model(batch)
        if sky is not None:
            outputs["sky"] = sky
            render.render_rgb_background = sky.render_rgb_background
            render.render_rgb_all_composited = (
                render.render_alpha_all * render.render_rgb_all
                + (1.0 - render.render_alpha_all) * sky.render_rgb_background
            )
        if profile_timing:
            outputs["timings_ms"] = timings_ms
        return outputs

    def _run_sky_model(self, batch: dict) -> SkyOutput | None:
        train_mode = str(batch.get("_train_mode", "v1-stable"))
        active_loss_weights = batch.get("_active_loss_weights", {})
        sky_alpha_active = isinstance(active_loss_weights, dict) and float(active_loss_weights.get("sky_alpha", 0.0)) > 0.0
        if (
            not bool(getattr(self.config, "render_enable_sky_background", True))
            and self.train_target != "sky_only"
            and train_mode != "v1-sky-ablation"
        ):
            return None
        if train_mode != "v1-sky-ablation" and self.train_target != "sky_only" and not sky_alpha_active:
            return None
        camera_intrinsics = batch.get("camera_intrinsics")
        camera_to_world = batch.get("camera_to_world")
        if camera_intrinsics is None or camera_to_world is None:
            return None
        sky_rgb = self.sky_model(
            camera_intrinsics=camera_intrinsics,
            camera_to_world=camera_to_world,
            image_size=(self.config.image_height, self.config.image_width),
        )
        return SkyOutput(
            render_rgb_background=sky_rgb,
            sky_mask_full=batch.get("sky_mask_full"),
        )

    def _build_source_view_candidate_mask(self, batch: dict, gaussians) -> torch.Tensor | None:
        source_view_ids = batch.get("source_view_ids")
        target_view_ids = batch.get("target_view_ids", batch.get("view_ids"))
        if source_view_ids is None or target_view_ids is None or "camera_to_world" not in batch:
            return None
        batch_size = gaussians.center.shape[0]
        source_views = gaussians.center.shape[2]
        target_views = batch["camera_to_world"].shape[2]
        mask = torch.ones(
            (batch_size, target_views, source_views),
            device=gaussians.center.device,
            dtype=gaussians.center.dtype,
        )
        for batch_idx in range(batch_size):
            src_ids = [int(view_id) for view_id in source_view_ids[batch_idx]]
            tgt_ids = [int(view_id) for view_id in target_view_ids[batch_idx]]
            src_index = {view_id: idx for idx, view_id in enumerate(src_ids[:source_views])}
            for target_idx, target_view_id in enumerate(tgt_ids[:target_views]):
                own_idx = src_index.get(target_view_id)
                if own_idx is None:
                    continue
                mask[batch_idx, target_idx].zero_()
                mask[batch_idx, target_idx, own_idx] = 1.0
        return mask

    def _run_bridges(self, gaussians, occ) -> BridgeOutput:
        gs_features = torch.cat([gaussians.instance_affinity, gaussians.motion_code], dim=-1)
        gs_to_occ_local, occ_to_gs_local = self.gs_occ_local_bridge(
            gaussians.center,
            gs_features,
            occ.occ_volumes["occ_1"],
        )

        b, t = gs_features.shape[:2]
        bt = b * t
        gs_proj = self.gs_token_proj(gs_features)
        gs_token_map = gs_proj.permute(0, 1, 5, 2, 3, 4).reshape(bt, gs_proj.shape[-1], *gs_proj.shape[2:5])
        gs_token_map = F.adaptive_avg_pool3d(gs_token_map, (min(gs_features.shape[2], 2), 7, 7))
        gs_tokens = gs_token_map.flatten(2).transpose(1, 2)

        occ_bridge_volume = occ.occ_volumes.get("occ_b", occ.occ_volumes["occ_1"])
        occ_tokens = F.adaptive_avg_pool3d(
            occ_bridge_volume.reshape(bt, occ_bridge_volume.shape[2], *occ_bridge_volume.shape[-3:]),
            (min(self.config.occ_grid.shape_zyx[0], 4), 10, 10),
        ).flatten(2).transpose(1, 2)
        occ_tokens = self.occ_latent_proj(occ_tokens)
        global_latents = self.gs_occ_global_latent_bridge(gs_tokens, occ_tokens).reshape(b, t, self.config.global_latents, -1)
        return BridgeOutput(
            gs_to_occ_local=gs_to_occ_local,
            occ_to_gs_local=occ_to_gs_local,
            global_latents=global_latents,
        )

    def _build_dummy_dynamic(self, xyz_1_4: torch.Tensor, dtype: torch.dtype) -> DynamicDenseOutput:
        b, t, v, _, h4, w4 = xyz_1_4.shape
        device = xyz_1_4.device
        dyn_logit_1_4 = torch.zeros((b, t, v, 1, h4, w4), device=device, dtype=dtype)
        dyn_logit_full = torch.zeros(
            (b, t, v, 1, self.config.image_height, self.config.image_width),
            device=device,
            dtype=dtype,
        )
        dyn_feat_1_4 = torch.zeros((b, t, v, self.config.dynamic_dense_dim, h4, w4), device=device, dtype=dtype)
        h2 = torch.zeros((b, t, v, self.config.neck_dim, h4, w4), device=device, dtype=dtype)
        p2 = torch.zeros_like(h2)
        full = torch.zeros(
            (b, t, v, self.config.full_dim, self.config.image_height, self.config.image_width),
            device=device,
            dtype=dtype,
        )
        return DynamicDenseOutput(
            dyn_logit_1_4=dyn_logit_1_4,
            dyn_logit_full=dyn_logit_full,
            dyn_feat_1_4=dyn_feat_1_4,
            h2=h2,
            p2=p2,
            full=full,
        )

    def _inject_bridge_context(self, entities: EntityOutput, bridges: BridgeOutput) -> EntityOutput:
        global_ctx = self.global_bridge_to_entity(bridges.global_latents.mean(dim=2))
        global_inst = global_ctx[..., : self.config.instance_dim][:, :, None, None, None, :]
        global_motion = global_ctx[..., self.config.instance_dim :][:, :, None, None, None, :]
        local_inst = bridges.occ_to_gs_local[..., : self.config.instance_dim]
        local_motion = bridges.occ_to_gs_local[..., self.config.instance_dim :]
        return EntityOutput(
            entity_id=entities.entity_id,
            refined_instance_affinity=entities.refined_instance_affinity + local_inst + global_inst,
            refined_motion_code=entities.refined_motion_code + local_motion + global_motion,
            gs_to_occ_feat=bridges.gs_to_occ_local,
        )

    def _build_dummy_assignment(self, gaussians) -> GaussianAssignmentOutput:
        b = gaussians.center.shape[0]
        num_gauss = gaussians.center.shape[1] * gaussians.center.shape[2] * gaussians.center.shape[3] * gaussians.center.shape[4]
        device = gaussians.center.device
        dtype = gaussians.center.dtype
        return GaussianAssignmentOutput(
            assignment_prob=torch.cat(
                [
                    torch.zeros((b, num_gauss, 1), device=device, dtype=dtype),
                    torch.ones((b, num_gauss, 1), device=device, dtype=dtype),
                ],
                dim=-1,
            ),
            assigned_query=torch.full((b, num_gauss), -1, device=device, dtype=torch.long),
            background_prob=torch.zeros((b, num_gauss), device=device, dtype=dtype),
            local_gate=torch.ones((b, num_gauss, 1), device=device, dtype=dtype),
            routing_keep_score=gaussians.keep_score.reshape(b, num_gauss, 1),
        )

    def _build_dummy_sem_proj(self, batch: dict, dtype: torch.dtype) -> torch.Tensor:
        b, t, v = batch["camera_to_world"].shape[:3]
        return torch.zeros(
            (b, t, v, self.config.projected_semantic_classes, self.config.image_height, self.config.image_width),
            device=batch["camera_to_world"].device,
            dtype=dtype,
        )

    @staticmethod
    def _downsample_points(points: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        b, t, v, h, w, c = points.shape
        x = points.permute(0, 1, 2, 5, 3, 4).reshape(b * t * v, c, h, w)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x.reshape(b, t, v, c, size[0], size[1])

    @staticmethod
    def _downsample_conf(conf: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        b, t, v, h, w = conf.shape
        x = conf.reshape(b * t * v, 1, h, w)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x.reshape(b, t, v, 1, size[0], size[1])
