"""Streaming object query memory pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .query_matching import match_queries_to_tracks


@dataclass
class QueryMemoryState:
    query_feat: torch.Tensor
    query_box3d: torch.Tensor
    query_motion: torch.Tensor
    query_cls_logit: torch.Tensor
    query_score: torch.Tensor
    query_id: torch.Tensor
    age: torch.Tensor
    miss: torch.Tensor
    active_mask: torch.Tensor


class QueryMemoryPool:
    def __init__(
        self,
        max_tracks: int = 64,
        feature_dim: int = 256,
        motion_dim: int = 7,
        class_dim: int = 8,
        max_miss: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.max_tracks = max_tracks
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        self.class_dim = class_dim
        self.max_miss = max_miss
        self.device = device
        self.state: Optional[QueryMemoryState] = None

    def reset(self, batch_size: int, device: Optional[torch.device] = None) -> QueryMemoryState:
        device = device or self.device or torch.device("cpu")
        self.state = QueryMemoryState(
            query_feat=torch.zeros(batch_size, self.max_tracks, self.feature_dim, device=device),
            query_box3d=torch.zeros(batch_size, self.max_tracks, 7, device=device),
            query_motion=torch.zeros(batch_size, self.max_tracks, self.motion_dim, device=device),
            query_cls_logit=torch.zeros(batch_size, self.max_tracks, self.class_dim, device=device),
            query_score=torch.zeros(batch_size, self.max_tracks, device=device),
            query_id=torch.full((batch_size, self.max_tracks), -1, dtype=torch.long, device=device),
            age=torch.zeros(batch_size, self.max_tracks, dtype=torch.long, device=device),
            miss=torch.zeros(batch_size, self.max_tracks, dtype=torch.long, device=device),
            active_mask=torch.zeros(batch_size, self.max_tracks, dtype=torch.bool, device=device),
        )
        return self.state

    def get(self, batch_size: int, device: torch.device) -> QueryMemoryState:
        if self.state is None or self.state.query_feat.shape[0] != batch_size:
            return self.reset(batch_size, device)
        return self.state

    def detach_state(self, state: QueryMemoryState) -> QueryMemoryState:
        return QueryMemoryState(
            query_feat=state.query_feat.detach(),
            query_box3d=state.query_box3d.detach(),
            query_motion=state.query_motion.detach(),
            query_cls_logit=state.query_cls_logit.detach(),
            query_score=state.query_score.detach(),
            query_id=state.query_id.detach(),
            age=state.age.detach(),
            miss=state.miss.detach(),
            active_mask=state.active_mask.detach(),
        )

    def update(self, state: QueryMemoryState) -> None:
        self.state = state

    def _pad_slots(self, tensor: torch.Tensor, pad_value: float | int | bool = 0) -> torch.Tensor:
        if tensor.shape[1] >= self.max_tracks:
            return tensor[:, : self.max_tracks]
        pad_shape = list(tensor.shape)
        pad_shape[1] = self.max_tracks - tensor.shape[1]
        pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)

    def training_update(
        self,
        prev_state: QueryMemoryState,
        query_feat: torch.Tensor,
        query_box3d: torch.Tensor,
        query_motion: torch.Tensor,
        query_cls_logit: torch.Tensor,
        query_score_logit: torch.Tensor,
        track_boxes_t: torch.Tensor,
        track_cls_t: torch.Tensor,
        track_visible_t: torch.Tensor,
        track_id_t: torch.Tensor,
        track_delta_t: Optional[torch.Tensor] = None,
        track_death_t: Optional[torch.Tensor] = None,
    ) -> QueryMemoryState:
        next_state = QueryMemoryState(
            query_feat=prev_state.query_feat.clone(),
            query_box3d=prev_state.query_box3d.clone(),
            query_motion=prev_state.query_motion.clone(),
            query_cls_logit=prev_state.query_cls_logit.clone(),
            query_score=prev_state.query_score.clone(),
            query_id=prev_state.query_id.clone(),
            age=prev_state.age.clone(),
            miss=prev_state.miss.clone(),
            active_mask=prev_state.active_mask.clone(),
        )
        track_boxes_t = self._pad_slots(track_boxes_t, 0.0)
        track_cls_t = self._pad_slots(track_cls_t, -1)
        track_visible_t = self._pad_slots(track_visible_t, False)
        track_id_t = self._pad_slots(track_id_t, -1)
        if track_delta_t is not None:
            track_delta_t = self._pad_slots(track_delta_t, 0.0)
        if track_death_t is not None:
            track_death_t = self._pad_slots(track_death_t, False)
        next_state.miss = next_state.miss + next_state.active_mask.long()
        updated_mask = torch.zeros_like(next_state.active_mask)

        for batch_idx in range(query_feat.shape[0]):
            match = match_queries_to_tracks(
                query_feat=query_feat[batch_idx],
                query_box3d=query_box3d[batch_idx],
                query_motion=query_motion[batch_idx],
                query_cls_logit=query_cls_logit[batch_idx],
                track_boxes_t=track_boxes_t[batch_idx, : self.max_tracks],
                track_cls_t=track_cls_t[batch_idx, : self.max_tracks],
                track_visible_t=track_visible_t[batch_idx, : self.max_tracks],
                track_id_t=track_id_t[batch_idx, : self.max_tracks],
                prev_query_feat=prev_state.query_feat[batch_idx],
                prev_query_id=prev_state.query_id[batch_idx],
                prev_active_mask=prev_state.active_mask[batch_idx],
                track_delta_t=None if track_delta_t is None else track_delta_t[batch_idx, : self.max_tracks],
            )
            visible_gt = torch.nonzero(match.gt_to_query >= 0, as_tuple=False).squeeze(-1)
            used_slots: set[int] = set()

            for gt_slot_tensor in visible_gt:
                gt_slot = int(gt_slot_tensor)
                query_slot = int(match.gt_to_query[gt_slot])
                prev_slot = int(match.gt_prev_slot[gt_slot])
                if prev_slot >= 0 and bool(prev_state.active_mask[batch_idx, prev_slot]) and prev_slot not in used_slots:
                    target_slot = prev_slot
                else:
                    free_slots = torch.nonzero(~next_state.active_mask[batch_idx], as_tuple=False).squeeze(-1)
                    free_slots = [int(slot) for slot in free_slots.tolist() if int(slot) not in used_slots]
                    if free_slots:
                        target_slot = free_slots[0]
                    else:
                        replace_scores = next_state.query_score[batch_idx].masked_fill(
                            updated_mask[batch_idx], float("inf")
                        )
                        target_slot = int(torch.argmin(replace_scores).item())
                used_slots.add(target_slot)
                updated_mask[batch_idx, target_slot] = True

                gt_box = track_boxes_t[batch_idx, gt_slot]
                gt_cls = track_cls_t[batch_idx, gt_slot]
                gt_id = track_id_t[batch_idx, gt_slot]
                gt_motion = query_motion[batch_idx, query_slot]
                if track_delta_t is not None:
                    gt_motion = track_delta_t[batch_idx, gt_slot]

                next_state.query_feat[batch_idx, target_slot] = query_feat[batch_idx, query_slot]
                next_state.query_box3d[batch_idx, target_slot] = gt_box
                next_state.query_motion[batch_idx, target_slot] = gt_motion
                cls_onehot = torch.zeros((self.class_dim,), dtype=query_cls_logit.dtype, device=query_cls_logit.device)
                if 0 <= int(gt_cls) < self.class_dim:
                    cls_onehot[int(gt_cls)] = 1.0
                next_state.query_cls_logit[batch_idx, target_slot] = cls_onehot
                next_state.query_score[batch_idx, target_slot] = torch.sigmoid(query_score_logit[batch_idx, query_slot])
                next_state.query_id[batch_idx, target_slot] = gt_id
                next_state.age[batch_idx, target_slot] = (
                    prev_state.age[batch_idx, target_slot] + 1 if bool(prev_state.active_mask[batch_idx, target_slot]) else 1
                )
                next_state.miss[batch_idx, target_slot] = 0
                next_state.active_mask[batch_idx, target_slot] = True

            death_track_ids: set[int] = set()
            if track_death_t is not None:
                death_slots = torch.nonzero(track_death_t[batch_idx, : self.max_tracks].bool(), as_tuple=False).squeeze(-1)
                for death_slot in death_slots.tolist():
                    death_track_ids.add(int(track_id_t[batch_idx, int(death_slot)]))

            for slot_idx in range(self.max_tracks):
                if bool(updated_mask[batch_idx, slot_idx]):
                    continue
                if not bool(prev_state.active_mask[batch_idx, slot_idx]):
                    next_state.active_mask[batch_idx, slot_idx] = False
                    next_state.query_score[batch_idx, slot_idx] = 0.0
                    next_state.query_id[batch_idx, slot_idx] = -1
                    continue
                track_id = int(prev_state.query_id[batch_idx, slot_idx])
                is_dead = track_id >= 0 and track_id in death_track_ids
                keep = (not is_dead) and int(next_state.miss[batch_idx, slot_idx]) <= self.max_miss
                next_state.active_mask[batch_idx, slot_idx] = keep
                if keep:
                    next_state.query_score[batch_idx, slot_idx] = next_state.query_score[batch_idx, slot_idx] * 0.95
                else:
                    next_state.query_score[batch_idx, slot_idx] = 0.0
                    next_state.query_id[batch_idx, slot_idx] = -1
        return next_state

    def inference_update(
        self,
        prev_state: QueryMemoryState,
        query_feat: torch.Tensor,
        query_box3d: torch.Tensor,
        query_motion: torch.Tensor,
        query_cls_logit: torch.Tensor,
        query_score_logit: torch.Tensor,
        keep_threshold: float = 0.35,
        birth_threshold: float = 0.55,
    ) -> QueryMemoryState:
        next_state = QueryMemoryState(
            query_feat=prev_state.query_feat.clone(),
            query_box3d=prev_state.query_box3d.clone(),
            query_motion=prev_state.query_motion.clone(),
            query_cls_logit=prev_state.query_cls_logit.clone(),
            query_score=prev_state.query_score.clone(),
            query_id=prev_state.query_id.clone(),
            age=prev_state.age.clone(),
            miss=prev_state.miss.clone(),
            active_mask=prev_state.active_mask.clone(),
        )

        track_score = torch.sigmoid(query_score_logit[:, : self.max_tracks])
        keep = track_score > keep_threshold
        next_state.query_feat = torch.where(keep.unsqueeze(-1), query_feat[:, : self.max_tracks], next_state.query_feat)
        next_state.query_box3d = torch.where(keep.unsqueeze(-1), query_box3d[:, : self.max_tracks], next_state.query_box3d)
        next_state.query_motion = torch.where(keep.unsqueeze(-1), query_motion[:, : self.max_tracks], next_state.query_motion)
        next_state.query_cls_logit = torch.where(keep.unsqueeze(-1), query_cls_logit[:, : self.max_tracks], next_state.query_cls_logit)
        next_state.query_score = torch.where(keep, track_score, next_state.query_score * 0.95)
        next_state.age = torch.where(keep, next_state.age + 1, next_state.age)
        next_state.miss = torch.where(keep, torch.zeros_like(next_state.miss), next_state.miss + next_state.active_mask.long())
        next_state.active_mask = keep | (next_state.active_mask & (next_state.miss <= self.max_miss))

        if query_feat.shape[1] > self.max_tracks:
            birth_feat = query_feat[:, self.max_tracks :]
            birth_box = query_box3d[:, self.max_tracks :]
            birth_motion = query_motion[:, self.max_tracks :]
            birth_cls = query_cls_logit[:, self.max_tracks :]
            birth_score = torch.sigmoid(query_score_logit[:, self.max_tracks :])
            free_mask = ~next_state.active_mask
            for batch_idx in range(query_feat.shape[0]):
                free_slots = torch.nonzero(free_mask[batch_idx], as_tuple=False).squeeze(-1)
                born = torch.nonzero(birth_score[batch_idx] > birth_threshold, as_tuple=False).squeeze(-1)
                if free_slots.numel() == 0 or born.numel() == 0:
                    continue
                born = born[: free_slots.numel()]
                slots = free_slots[: born.numel()]
                next_state.query_feat[batch_idx, slots] = birth_feat[batch_idx, born]
                next_state.query_box3d[batch_idx, slots] = birth_box[batch_idx, born]
                next_state.query_motion[batch_idx, slots] = birth_motion[batch_idx, born]
                next_state.query_cls_logit[batch_idx, slots] = birth_cls[batch_idx, born]
                next_state.query_score[batch_idx, slots] = birth_score[batch_idx, born]
                next_state.query_id[batch_idx, slots] = -1
                next_state.age[batch_idx, slots] = 1
                next_state.miss[batch_idx, slots] = 0
                next_state.active_mask[batch_idx, slots] = True

        next_state.query_score = torch.where(next_state.active_mask, next_state.query_score, torch.zeros_like(next_state.query_score))
        next_state.query_id = torch.where(next_state.active_mask, next_state.query_id, torch.full_like(next_state.query_id, -1))
        return next_state
