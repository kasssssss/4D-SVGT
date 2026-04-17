"""Top-level scaffold that wires the first model interfaces together."""

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
from dvgt_occ.types import BridgeOutput, EntityOutput


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
        self.gs_head = GSHead(channels=config.neck_dim, instance_dim=config.instance_dim, motion_dim=config.motion_dim)
        self.gaussian_merge = GaussianMultiViewMerge()
        self.gaussian_assignment = GaussianQueryAssignment()
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
        )
        self.mask_renderer = GaussianMaskRenderer(output_size=(config.image_height, config.image_width))

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.dynamic_dense.set_gradient_checkpointing(enabled)
        self.occ_head.set_gradient_checkpointing(enabled)
        self.gs_head.set_gradient_checkpointing(enabled)

    def forward(self, batch: dict) -> dict:
        features = self.reassembly(batch["aggregated_tokens"])
        points = batch["points"]
        points_conf = batch["points_conf"]
        xyz_1_4 = self._downsample_points(points, size=(56, 112))
        conf_1_4 = self._downsample_conf(points_conf, size=(56, 112))
        xyz_1_8 = self._downsample_points(points, size=(28, 56))
        dynamic = self.dynamic_dense(features, xyz_1_4=xyz_1_4, conf_1_4=conf_1_4)
        occ = self.occ_head(features, dynamic=dynamic, points=points, points_conf=points_conf)
        queries = self.dynamic_query(
            occ.occ_volumes,
            dynamic_feat_1_4=dynamic.dyn_feat_1_4,
            dynamic_logit_1_4=dynamic.dyn_logit_1_4,
            dynamic_xyz_1_4=xyz_1_4,
            batch=batch,
        )
        gaussians = self.gs_head(features, dynamic=dynamic, xyz_1_8=xyz_1_8)
        global_track_id = batch.get("gs_global_track_id_1_8")
        gaussians_merged = self.gaussian_merge(gaussians, global_track_id=global_track_id)
        assignment = self.gaussian_assignment(gaussians_merged, queries)
        bridges = self._run_bridges(gaussians_merged, occ)
        entities = self.entity_aggregator(gaussians_merged, queries, assignment)
        entities = self._inject_bridge_context(entities, bridges)
        sem_proj_2d = self.occ_semantic_projector(occ.occ_logit, occ.sem_logit, num_views=self.config.num_views)
        render = self.mask_renderer(gaussians_merged, assignment, sem_proj_2d)
        return {
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
