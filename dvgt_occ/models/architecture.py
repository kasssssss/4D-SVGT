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
from dvgt_occ.types import BridgeOutput, EntityOutput, GaussianAssignmentOutput, SkyOutput


class DVGTOccModel(nn.Module):
    def __init__(self, config: DVGTOccConfig = DEFAULT_DVGT_OCC_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.reassembly = TokenReassembly(
            selected_layers=config.selected_layers,
            in_dim=config.agg_token_dim,
            out_dim=config.neck_dim,
            patch_grid=config.patch_grid,
        )
        self.dynamic_dense = DynamicDenseBranch(channels=config.neck_dim, full_channels=config.full_dim)
        self.occ_head = OccHead(
            channels=config.neck_dim,
            semantic_classes=config.semantic_classes,
            occ_shape_zyx=config.occ_grid.shape_zyx,
            x_range=config.occ_grid.x_range,
            y_range=config.occ_grid.y_range,
            z_range=config.occ_grid.z_range,
            voxel_size=config.occ_grid.voxel_size,
        )
        self.dynamic_query = DynamicQueryDecoder(
            query_dim=config.dynamic_query_dim,
            max_track_queries=config.max_track_queries,
            new_queries=config.new_queries,
            dynamic_classes=config.dynamic_classes,
            motion_dim=7,
            local_feature_dim=config.neck_dim,
            local_topk=min(config.sparse_dynamic_anchors, config.num_views * 56 * 112),
            occ_samples_per_scale=config.occ_samples_per_scale,
            x_range=config.occ_grid.x_range,
            y_range=config.occ_grid.y_range,
            z_range=config.occ_grid.z_range,
        )
        self.gs_head = GSHead(
            channels=config.neck_dim,
            instance_dim=config.instance_dim,
            motion_dim=config.motion_dim,
            bias_scale=config.gs_bias_scale,
        )
        self.gaussian_merge = GaussianMultiViewMerge()
        self.gaussian_assignment = GaussianQueryAssignment(
            query_dim=config.dynamic_query_dim,
            gaussian_feat_dim=config.instance_dim + config.motion_dim,
        )
        self.gs_occ_local_bridge = GSOccLocalBridge(
            occ_channels=config.neck_dim,
            gs_channels=config.instance_dim + config.motion_dim,
            grid_min_xyz=(config.occ_grid.x_range[0], config.occ_grid.y_range[0], config.occ_grid.z_range[0]),
            voxel_size=config.occ_grid.voxel_size,
            grid_shape_zyx=config.occ_grid.shape_zyx,
        )
        self.gs_occ_global_latent_bridge = GSOccGlobalLatentBridge(channels=config.neck_dim, num_latents=config.global_latents)
        self.gs_token_proj = nn.Linear(config.instance_dim + config.motion_dim, config.neck_dim)
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

        with torch.autograd.profiler.record_function("dvgt_occ/reassembly"):
            features = _record_timing("reassembly", lambda: self.reassembly(batch["aggregated_tokens"]))
        points = batch["points"]
        points_conf = batch["points_conf"]
        xyz_1_4 = self._downsample_points(points, size=(56, 112))
        conf_1_4 = self._downsample_conf(points_conf, size=(56, 112))
        xyz_1_8 = self._downsample_points(points, size=(28, 56))
        with torch.autograd.profiler.record_function("dvgt_occ/dynamic_dense"):
            dynamic = _record_timing(
                "dynamic_dense",
                lambda: self.dynamic_dense(features, xyz_1_4=xyz_1_4, conf_1_4=conf_1_4),
            )
        if self.train_target == "occ_only":
            with torch.autograd.profiler.record_function("dvgt_occ/occ_head"):
                occ = _record_timing("occ_head", lambda: self.occ_head(features, dynamic=dynamic, points=points, points_conf=points_conf))
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
                gaussians = _record_timing("gs_head", lambda: self.gs_head(features, dynamic=dynamic, xyz_1_8=xyz_1_8))
            with torch.autograd.profiler.record_function("dvgt_occ/gaussian_merge"):
                gaussians_merged = _record_timing("gaussian_merge", lambda: self.gaussian_merge(gaussians, global_track_id=None))
            dummy_assignment = self._build_dummy_assignment(gaussians_merged)
            dummy_sem_proj_2d = self._build_dummy_sem_proj(batch, gaussians_merged.center.dtype)
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
            occ = _record_timing("occ_head", lambda: self.occ_head(features, dynamic=dynamic, points=points, points_conf=points_conf))
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
            gaussians = _record_timing("gs_head", lambda: self.gs_head(features, dynamic=dynamic, xyz_1_8=xyz_1_8))
        global_track_id = batch.get("gs_global_track_id_1_8")
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

        occ_tokens = F.adaptive_avg_pool3d(
            occ.occ_volumes["occ_1"].reshape(bt, occ.occ_volumes["occ_1"].shape[2], *occ.occ_volumes["occ_1"].shape[-3:]),
            (min(self.config.occ_grid.shape_zyx[0], 4), 10, 10),
        ).flatten(2).transpose(1, 2)
        global_latents = self.gs_occ_global_latent_bridge(gs_tokens, occ_tokens).reshape(b, t, self.config.global_latents, -1)
        return BridgeOutput(
            gs_to_occ_local=gs_to_occ_local,
            occ_to_gs_local=occ_to_gs_local,
            global_latents=global_latents,
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
