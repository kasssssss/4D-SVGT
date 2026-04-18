"""Query/track matching helpers shared by training losses and memory routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback for minimal envs
    np = None
    linear_sum_assignment = None

INVALID_COST = 1e6


@dataclass
class QueryTrackMatchResult:
    query_to_gt: torch.Tensor
    query_to_track_id: torch.Tensor
    query_match_cost: torch.Tensor
    gt_to_query: torch.Tensor
    gt_prev_slot: torch.Tensor
    gt_is_birth: torch.Tensor


def _hungarian_assignment(cost: list[list[float]]) -> list[int]:
    """Return the assigned column index for each row."""

    if not cost:
        return []
    rows = len(cost)
    cols = len(cost[0]) if cost[0] else 0
    if cols == 0:
        return [-1] * rows
    if rows > cols:
        raise ValueError("Hungarian solver expects rows <= cols.")

    if linear_sum_assignment is not None and np is not None:
        matrix = np.asarray(cost, dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=INVALID_COST, posinf=INVALID_COST, neginf=INVALID_COST)
        row_ind, col_ind = linear_sum_assignment(matrix)
        assignment = [-1] * rows
        for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
            assignment[int(row_idx)] = int(col_idx)
        return assignment

    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)

    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * rows
    for j in range(1, cols + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def match_queries_to_tracks(
    query_feat: torch.Tensor,
    query_box3d: torch.Tensor,
    query_motion: torch.Tensor,
    query_cls_logit: torch.Tensor,
    track_boxes_t: torch.Tensor,
    track_cls_t: torch.Tensor,
    track_visible_t: torch.Tensor,
    track_id_t: torch.Tensor,
    prev_query_feat: Optional[torch.Tensor] = None,
    prev_query_id: Optional[torch.Tensor] = None,
    prev_active_mask: Optional[torch.Tensor] = None,
    track_delta_t: Optional[torch.Tensor] = None,
) -> QueryTrackMatchResult:
    device = query_box3d.device
    num_queries = query_box3d.shape[0]
    num_tracks = track_boxes_t.shape[0]
    query_to_gt = torch.full((num_queries,), -1, dtype=torch.long, device=device)
    query_to_track_id = torch.full((num_queries,), -1, dtype=torch.long, device=device)
    query_match_cost = torch.zeros((num_queries,), dtype=query_box3d.dtype, device=device)
    gt_to_query = torch.full((num_tracks,), -1, dtype=torch.long, device=device)
    gt_prev_slot = torch.full((num_tracks,), -1, dtype=torch.long, device=device)
    gt_is_birth = torch.zeros((num_tracks,), dtype=torch.bool, device=device)

    visible_idx = torch.nonzero(track_visible_t.bool(), as_tuple=False).squeeze(-1)
    if visible_idx.numel() == 0 or num_queries == 0:
        return QueryTrackMatchResult(
            query_to_gt=query_to_gt,
            query_to_track_id=query_to_track_id,
            query_match_cost=query_match_cost,
            gt_to_query=gt_to_query,
            gt_prev_slot=gt_prev_slot,
            gt_is_birth=gt_is_birth,
        )

    gt_boxes = track_boxes_t[visible_idx]
    gt_cls = track_cls_t[visible_idx].clamp(min=0)
    gt_ids = track_id_t[visible_idx]

    center_cost = torch.cdist(gt_boxes[:, :3], query_box3d[:, :3], p=2) / 10.0
    size_cost = (gt_boxes[:, None, 3:6] - query_box3d[None, :, 3:6]).abs().mean(dim=-1)

    cls_prob = torch.softmax(query_cls_logit, dim=-1)
    cls_cost = 1.0 - cls_prob[:, gt_cls].transpose(0, 1)

    motion_cost = torch.zeros_like(center_cost)
    if track_delta_t is not None and track_delta_t.numel() > 0:
        gt_motion = track_delta_t[visible_idx][..., : query_motion.shape[-1]]
        motion_cost = (gt_motion[:, None, :] - query_motion[None, :, : gt_motion.shape[-1]]).abs().mean(dim=-1)

    feature_cost = torch.zeros_like(center_cost)
    if prev_query_feat is not None and prev_query_id is not None and prev_active_mask is not None:
        norm_query_feat = F.normalize(query_feat, dim=-1)
        norm_prev_feat = F.normalize(prev_query_feat, dim=-1)
        for row_idx, track_id in enumerate(gt_ids.tolist()):
            prev_slots = torch.nonzero(prev_active_mask & (prev_query_id == int(track_id)), as_tuple=False).squeeze(-1)
            if prev_slots.numel() == 0:
                continue
            prev_slot = int(prev_slots[0])
            gt_prev_slot[visible_idx[row_idx]] = prev_slot
            feature_cost[row_idx] = 0.5 * (1.0 - torch.matmul(norm_query_feat, norm_prev_feat[prev_slot]))

    total_cost = (
        0.45 * center_cost
        + 0.20 * size_cost
        + 0.20 * cls_cost
        + 0.10 * motion_cost
        + 0.05 * feature_cost
    )
    total_cost = torch.nan_to_num(
        total_cost,
        nan=INVALID_COST,
        posinf=INVALID_COST,
        neginf=INVALID_COST,
    )

    assignment = _hungarian_assignment(total_cost.detach().cpu().tolist())
    for row_idx, query_idx in enumerate(assignment):
        if query_idx < 0:
            continue
        gt_slot = int(visible_idx[row_idx])
        query_to_gt[query_idx] = gt_slot
        query_to_track_id[query_idx] = int(track_id_t[gt_slot])
        query_match_cost[query_idx] = total_cost[row_idx, query_idx]
        gt_to_query[gt_slot] = query_idx
        gt_is_birth[gt_slot] = gt_prev_slot[gt_slot] < 0

    return QueryTrackMatchResult(
        query_to_gt=query_to_gt,
        query_to_track_id=query_to_track_id,
        query_match_cost=query_match_cost,
        gt_to_query=gt_to_query,
        gt_prev_slot=gt_prev_slot,
        gt_is_birth=gt_is_birth,
    )
