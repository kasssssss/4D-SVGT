"""Occ-first dynamic query decoder with streaming memory."""

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dvgt_occ.types import DynamicQueryOutput

from .deformable_3d_attention import MultiScaleDeformable3DAttention
from .query_memory_pool import QueryMemoryPool, QueryMemoryState


class DynamicQueryDecoder(nn.Module):
    def __init__(
        self,
        query_dim: int = 256,
        max_track_queries: int = 64,
        new_queries: int = 64,
        dynamic_classes: int = 8,
        motion_dim: int = 7,
        local_feature_dim: int = 256,
        local_topk: int = 2048,
        occ_samples_per_scale: int = 8,
        x_range: tuple[float, float] = (-40.0, 40.0),
        y_range: tuple[float, float] = (-40.0, 40.0),
        z_range: tuple[float, float] = (-2.0, 6.0),
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.max_track_queries = max_track_queries
        self.motion_dim = motion_dim
        self.local_topk = local_topk
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self.new_query_embed = nn.Embedding(new_queries, query_dim)
        self.track_feat_proj = nn.Linear(query_dim, query_dim)
        self.track_box_proj = nn.Linear(7, query_dim)
        self.track_motion_proj = nn.Linear(motion_dim, query_dim)
        self.track_cls_proj = nn.Linear(dynamic_classes, query_dim)
        self.track_score_proj = nn.Linear(1, query_dim)

        self.pre_norm = nn.LayerNorm(query_dim)
        self.self_attn = nn.MultiheadAttention(query_dim, num_heads=8, batch_first=True)
        self.ref_point_head = nn.Linear(query_dim, 3)
        self.attn = MultiScaleDeformable3DAttention(
            query_dim=query_dim,
            samples_per_scale=occ_samples_per_scale,
            volume_channels=(64, 128, 128),
        )
        self.local_token_proj = nn.Linear(local_feature_dim + 3, query_dim)
        self.local_attn = nn.MultiheadAttention(query_dim, num_heads=8, batch_first=True)
        self.ffn_norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(nn.Linear(query_dim, query_dim * 4), nn.GELU(), nn.Linear(query_dim * 4, query_dim))

        self.presence_head = nn.Linear(query_dim, 1)
        self.box_head = nn.Linear(query_dim, 7)
        self.cls_head = nn.Linear(query_dim, dynamic_classes)
        self.motion_head = nn.Linear(query_dim, motion_dim)
        self.memory_pool = QueryMemoryPool(
            max_tracks=max_track_queries,
            feature_dim=query_dim,
            motion_dim=motion_dim,
            class_dim=dynamic_classes,
        )

    def forward(
        self,
        occ_volumes: Mapping[str, torch.Tensor],
        dynamic_feat_1_4: torch.Tensor,
        dynamic_logit_1_4: torch.Tensor,
        dynamic_xyz_1_4: torch.Tensor,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        memory_state: Optional[QueryMemoryState] = None,
    ) -> DynamicQueryOutput:
        device = next(self.parameters()).device
        batch_size = dynamic_feat_1_4.shape[0]
        num_frames = dynamic_feat_1_4.shape[1]
        state = memory_state if memory_state is not None else self.memory_pool.reset(batch_size, device=device)

        presence_steps = []
        box_steps = []
        cls_steps = []
        feat_steps = []
        motion_steps = []
        ref_steps = []
        mask_steps = []

        new_queries = self.new_query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        for time_idx in range(num_frames):
            track_queries, propagated_boxes = self._build_track_queries(state)
            queries = torch.cat([track_queries, new_queries], dim=1)
            queries = queries + self.self_attn(self.pre_norm(queries), self.pre_norm(queries), self.pre_norm(queries), need_weights=False)[0]
            refs = self._build_reference_points(queries, state.active_mask, propagated_boxes)
            occ_step = {key: value[:, time_idx] for key, value in occ_volumes.items()}
            queries = queries + self.attn(self.pre_norm(queries), refs, occ_step)

            local_tokens = self._build_local_tokens(
                dynamic_feat_1_4[:, time_idx],
                dynamic_logit_1_4[:, time_idx],
                dynamic_xyz_1_4[:, time_idx],
            )
            queries = queries + self.local_attn(self.pre_norm(queries), local_tokens, local_tokens, need_weights=False)[0]
            queries = queries + self.ffn(self.ffn_norm(queries))

            presence = self.presence_head(queries).squeeze(-1)
            box_delta = self.box_head(queries)
            box = box_delta.clone()
            box[:, : self.max_track_queries] = box[:, : self.max_track_queries] + propagated_boxes
            cls = self.cls_head(queries)
            motion = self.motion_head(queries)

            presence_steps.append(presence)
            box_steps.append(box)
            cls_steps.append(cls)
            feat_steps.append(queries)
            motion_steps.append(motion)
            ref_steps.append(refs)
            mask_steps.append(state.active_mask)

            if batch is not None and "track_boxes_3d" in batch:
                track_delta_t = None
                if "track_delta_box" in batch and time_idx < batch["track_delta_box"].shape[1]:
                    track_delta_t = batch["track_delta_box"][:, time_idx]
                track_death_t = None
                if "track_death" in batch and time_idx < batch["track_death"].shape[1]:
                    track_death_t = batch["track_death"][:, time_idx]
                state = self.memory_pool.training_update(
                    prev_state=state,
                    query_feat=queries,
                    query_box3d=box,
                    query_motion=motion,
                    query_cls_logit=cls,
                    query_score_logit=presence,
                    track_boxes_t=batch["track_boxes_3d"][:, time_idx],
                    track_cls_t=batch["track_cls"][:, time_idx],
                    track_visible_t=batch["track_visible"][:, time_idx],
                    track_id_t=batch["track_id"][:, time_idx],
                    track_delta_t=track_delta_t,
                    track_death_t=track_death_t,
                )
            else:
                state = self.memory_pool.inference_update(
                    prev_state=state,
                    query_feat=queries,
                    query_box3d=box,
                    query_motion=motion,
                    query_cls_logit=cls,
                    query_score_logit=presence,
                )

        self.memory_pool.update(self.memory_pool.detach_state(state))
        return DynamicQueryOutput(
            presence_logit=torch.stack(presence_steps, dim=1),
            query_box3d=torch.stack(box_steps, dim=1),
            query_cls_logit=torch.stack(cls_steps, dim=1),
            query_feat=torch.stack(feat_steps, dim=1),
            query_motion=torch.stack(motion_steps, dim=1),
            query_ref_points=torch.stack(ref_steps, dim=1),
            track_query_mask=torch.stack(mask_steps, dim=1),
        )

    def _build_track_queries(self, state: QueryMemoryState) -> tuple[torch.Tensor, torch.Tensor]:
        propagated_boxes = state.query_box3d.clone()
        motion_dim = min(state.query_motion.shape[-1], propagated_boxes.shape[-1])
        propagated_boxes[..., :motion_dim] = propagated_boxes[..., :motion_dim] + state.query_motion[..., :motion_dim]
        queries = (
            self.track_feat_proj(state.query_feat)
            + self.track_box_proj(propagated_boxes)
            + self.track_motion_proj(state.query_motion)
            + self.track_cls_proj(state.query_cls_logit)
            + self.track_score_proj(state.query_score.unsqueeze(-1))
        )
        queries = F.layer_norm(queries, (self.query_dim,))
        queries = queries * state.active_mask.unsqueeze(-1).to(queries.dtype)
        return queries, propagated_boxes

    def _build_reference_points(
        self,
        queries: torch.Tensor,
        active_mask: torch.Tensor,
        propagated_boxes: torch.Tensor,
    ) -> torch.Tensor:
        refs = torch.sigmoid(self.ref_point_head(queries))
        box_refs = self._normalize_xyz(propagated_boxes[..., :3])
        track_refs = torch.where(active_mask.unsqueeze(-1), box_refs, refs[:, : self.max_track_queries])
        return torch.cat([track_refs, refs[:, self.max_track_queries :]], dim=1).clamp(0.0, 1.0)

    def _normalize_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        x = (xyz[..., 0] - self.x_range[0]) / max(self.x_range[1] - self.x_range[0], 1e-6)
        y = (xyz[..., 1] - self.y_range[0]) / max(self.y_range[1] - self.y_range[0], 1e-6)
        z = (xyz[..., 2] - self.z_range[0]) / max(self.z_range[1] - self.z_range[0], 1e-6)
        return torch.stack([x, y, z], dim=-1)

    def _build_local_tokens(
        self,
        dynamic_feat_t: torch.Tensor,
        dynamic_logit_t: torch.Tensor,
        dynamic_xyz_t: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_views, channels, height, width = dynamic_feat_t.shape
        feat_flat = dynamic_feat_t.permute(0, 1, 3, 4, 2).reshape(batch_size, num_views * height * width, channels)
        xyz_flat = dynamic_xyz_t.permute(0, 1, 3, 4, 2).reshape(batch_size, num_views * height * width, 3)
        score_flat = dynamic_logit_t.squeeze(2).reshape(batch_size, num_views * height * width)
        topk = min(self.local_topk, feat_flat.shape[1])
        topk_idx = torch.topk(score_flat, k=topk, dim=1).indices
        gather_feat = torch.gather(feat_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, channels))
        gather_xyz = torch.gather(xyz_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 3))
        gather_score = torch.gather(torch.sigmoid(score_flat), 1, topk_idx).unsqueeze(-1)
        return self.local_token_proj(torch.cat([gather_feat * gather_score, gather_xyz], dim=-1))
