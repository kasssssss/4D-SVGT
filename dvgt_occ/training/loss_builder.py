"""Stage-B loss builder for small-sample overfit and early stable training."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dvgt_occ.config import DVGTOccConfig
from dvgt_occ.models.dynamic import match_queries_to_tracks
from dvgt_occ.training.stage_scheduler import DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS, LossWeights


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _safe_mean(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return _zero_like(reference)
    return values.mean()


def _balanced_bce_with_logits(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    pos = target.sum()
    neg = float(target.numel()) - float(pos.item())
    if pos.item() <= 0:
        return F.binary_cross_entropy_with_logits(pred, target)
    pos_weight = pred.new_tensor([min(max(neg / float(pos.item()), 1.0), 20.0)])
    return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)


def _soft_dice_loss_from_logits(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target.float()
    prob = torch.sigmoid(pred)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _safe_logit(prob: torch.Tensor) -> torch.Tensor:
    return torch.logit(prob.clamp(1e-4, 1.0 - 1e-4))


class DVGTOccLossBuilder(nn.Module):
    def __init__(
        self,
        config: DVGTOccConfig,
        weights: LossWeights = DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS,
    ) -> None:
        super().__init__()
        self.config = config
        self.weights = weights

    def forward(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reference = outputs["dynamic"].dyn_logit_1_4
        losses: Dict[str, torch.Tensor] = {}

        losses["dyn_soft"] = self._loss_dyn_soft(outputs, batch)
        losses["presence"], losses["query_match"], losses["query_cls"], losses["query_box3d"] = self._loss_queries(outputs, batch)
        losses["track_mem_feat"] = self._loss_entity_motion(outputs, batch)
        losses["track_mem_box"] = self._loss_query_motion(outputs, batch)
        losses["birth_death"] = self._loss_birth_death(outputs, batch)
        losses["occ"] = self._loss_occ(outputs, batch)
        losses["sem_occ"] = self._loss_sem_occ(outputs, batch)
        losses["dyn_occ"] = self._loss_dyn_occ(outputs, batch)
        losses["inst_contrast"] = self._loss_instance_contrast(outputs, batch)
        losses["query2gs"], losses["q2gs_null"] = self._loss_query_to_gaussian(outputs, batch)
        losses["mask_render"] = self._loss_dyn_mask_full(outputs, batch)
        losses["occ_local2gs"] = self._loss_occ_local2gs(outputs, batch)
        losses["gs2occ_local"] = self._loss_gs2occ_local(outputs, batch)
        losses["sem_proj_2d"] = self._loss_sem_proj_2d(outputs, batch)
        losses["motion_cons"] = self._loss_motion_cons(outputs, batch)
        losses["gs_sparse"] = self._loss_gs_sparse(outputs, batch)

        total = _zero_like(reference)
        for key, weight in asdict(self.weights).items():
            total = total + float(weight) * losses[key]
        return total, losses

    def _loss_dyn_soft(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = outputs["dynamic"].dyn_logit_1_4.squeeze(3)
        target = batch["dyn_soft_mask_1_4"]
        return _balanced_bce_with_logits(pred, target) + _soft_dice_loss_from_logits(pred, target)

    def _loss_dyn_mask_full(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "sam3_dyn_mask_full" in batch and "render" in outputs:
            pred_dyn = _safe_logit(outputs["render"].render_alpha_dynamic.squeeze(3))
            target_dyn = batch["sam3_dyn_mask_full"]
            loss_dyn = _balanced_bce_with_logits(pred_dyn, target_dyn) + _soft_dice_loss_from_logits(pred_dyn, target_dyn)
            pred_all = _safe_logit(outputs["render"].render_alpha_all.squeeze(3))
            target_all = batch.get("sam3_all_mask_full", target_dyn)
            loss_all = _balanced_bce_with_logits(pred_all, target_all) + _soft_dice_loss_from_logits(pred_all, target_all)
            return loss_dyn + 0.25 * loss_all

        pred = outputs["dynamic"].dyn_logit_full.squeeze(3)
        target = batch["dyn_soft_mask_1_4"]
        b, t, v, h, w = target.shape
        target_full = F.interpolate(
            target.reshape(b * t * v, 1, h, w),
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, v, pred.shape[-2], pred.shape[-1])
        return _balanced_bce_with_logits(pred, target_full) + _soft_dice_loss_from_logits(pred, target_full)

    def _loss_occ(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = outputs["occ"].occ_logit.squeeze(2)
        target = batch["occ_label"]
        return F.binary_cross_entropy_with_logits(pred, target)

    def _loss_sem_occ(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = outputs["occ"].sem_logit
        b, t, c, z, y, x = pred.shape
        target = batch["occ_sem_label"]
        return F.cross_entropy(pred.reshape(b * t, c, z, y, x), target.reshape(b * t, z, y, x))

    def _loss_dyn_occ(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = outputs["occ"].dyn_occ_logit.squeeze(2)
        target = batch["occ_dyn_label"]
        return F.binary_cross_entropy_with_logits(pred, target)

    def _loss_queries(
        self,
        outputs: Dict[str, object],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = outputs["queries"]
        matched_gt_slot, cls_target, box_target, visible_mask, match_cost = self._build_query_targets_hungarian(
            batch=batch,
            query_feat=queries.query_feat,
            query_box3d=queries.query_box3d,
            query_motion=queries.query_motion,
            query_cls_logit=queries.query_cls_logit,
        )
        presence_target = (matched_gt_slot >= 0).float()

        loss_presence = _balanced_bce_with_logits(queries.presence_logit, presence_target)
        loss_match = _safe_mean(match_cost[visible_mask], queries.presence_logit)

        if torch.any(visible_mask):
            loss_cls = F.cross_entropy(queries.query_cls_logit[visible_mask], cls_target[visible_mask])
            loss_box = F.smooth_l1_loss(queries.query_box3d[visible_mask], box_target[visible_mask])
        else:
            loss_cls = _zero_like(queries.presence_logit)
            loss_box = _zero_like(queries.presence_logit)
        return loss_presence, loss_match, loss_cls, loss_box

    def _loss_instance_contrast(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = outputs["entities"].refined_instance_affinity
        global_ids = batch["gs_global_track_id_1_8"]
        reference = embeddings
        losses = []
        for batch_idx in range(embeddings.shape[0]):
            emb_flat = F.normalize(embeddings[batch_idx].reshape(-1, embeddings.shape[-1]), dim=-1)
            id_flat = global_ids[batch_idx].reshape(-1)
            valid_ids = torch.unique(id_flat[id_flat >= 0])
            if valid_ids.numel() == 0:
                continue
            prototypes = []
            pos_parts = []
            for track_id in valid_ids.tolist():
                mask = id_flat == track_id
                proto = F.normalize(emb_flat[mask].mean(dim=0, keepdim=True), dim=-1)
                prototypes.append(proto.squeeze(0))
                pos_parts.append((1.0 - (emb_flat[mask] * proto).sum(dim=-1)).mean())
            pos_loss = torch.stack(pos_parts).mean()
            if len(prototypes) > 1:
                proto_stack = torch.stack(prototypes, dim=0)
                sim = proto_stack @ proto_stack.t()
                eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
                neg_loss = F.relu(sim[~eye] - 0.2).mean()
            else:
                neg_loss = _zero_like(reference)
            losses.append(pos_loss + neg_loss)
        if not losses:
            return _zero_like(reference)
        return torch.stack(losses).mean()

    def _loss_query_motion(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        queries = outputs["queries"]
        _, _, motion_target, visible_mask, _ = self._build_query_targets_hungarian(
            batch=batch,
            query_feat=queries.query_feat,
            query_box3d=queries.query_box3d,
            query_motion=queries.query_motion,
            query_cls_logit=queries.query_cls_logit,
            box_source=batch["track_delta_box"],
        )
        query_motion = queries.query_motion[:, : motion_target.shape[1]]
        if torch.any(visible_mask):
            return F.smooth_l1_loss(query_motion[visible_mask], motion_target[visible_mask])
        return _zero_like(query_motion)

    def _loss_birth_death(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        queries = outputs["queries"]
        matched_gt_slot, _, _, visible_mask, _ = self._build_query_targets_hungarian(
            batch=batch,
            query_feat=queries.query_feat,
            query_box3d=queries.query_box3d,
            query_motion=queries.query_motion,
            query_cls_logit=queries.query_cls_logit,
        )
        birth_target = batch["track_birth"]
        if birth_target.numel() == 0:
            return _zero_like(queries.presence_logit)

        losses = []
        for batch_idx in range(queries.presence_logit.shape[0]):
            for time_idx in range(queries.presence_logit.shape[1]):
                matched = torch.nonzero(visible_mask[batch_idx, time_idx], as_tuple=False).squeeze(-1)
                if matched.numel() == 0:
                    continue
                gt_slot = matched_gt_slot[batch_idx, time_idx, matched].long().clamp(min=0)
                birth_mask = birth_target[batch_idx, time_idx, gt_slot].float()
                if birth_mask.numel() == 0:
                    continue
                pred = queries.presence_logit[batch_idx, time_idx, matched]
                losses.append(F.binary_cross_entropy_with_logits(pred, birth_mask))
        if not losses:
            return _zero_like(queries.presence_logit)
        return torch.stack(losses).mean()

    def _loss_entity_motion(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        refined_motion = outputs["entities"].refined_motion_code[..., :7]
        global_ids = batch["gs_global_track_id_1_8"]
        track_delta = batch["track_delta_box"]
        track_id = batch["track_id"]
        track_visible = batch["track_visible"]
        reference = refined_motion
        losses = []
        for batch_idx in range(refined_motion.shape[0]):
            max_slots = min(track_id.shape[2], self.config.max_track_queries)
            for time_idx in range(refined_motion.shape[1]):
                if time_idx >= track_delta.shape[1]:
                    continue
                motion_flat = refined_motion[batch_idx, time_idx].reshape(-1, refined_motion.shape[-1])
                id_flat = global_ids[batch_idx, time_idx].reshape(-1)
                for slot_idx in range(max_slots):
                    if not bool(track_visible[batch_idx, time_idx, slot_idx]):
                        continue
                    track_global_id = int(track_id[batch_idx, time_idx, slot_idx])
                    mask = id_flat == track_global_id
                    if not torch.any(mask):
                        continue
                    pooled_motion = motion_flat[mask].mean(dim=0)
                    losses.append(F.smooth_l1_loss(pooled_motion, track_delta[batch_idx, time_idx, slot_idx]))

        if not losses:
            return _zero_like(reference)
        return torch.stack(losses).mean()

    def _loss_query_to_gaussian(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        assignment = outputs["assignment"]
        target_slot = self._build_gaussian_query_targets(batch, self.config.max_track_queries)
        fg_mask = target_slot >= 0
        bg_mask = target_slot < 0

        fg_loss = _zero_like(assignment.assignment_prob)
        if torch.any(fg_mask):
            fg_prob = assignment.assignment_prob[..., 1:].gather(-1, target_slot.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            fg_loss = -torch.log(fg_prob[fg_mask].clamp_min(1e-6)).mean()

        bg_loss = _zero_like(assignment.assignment_prob)
        if torch.any(bg_mask):
            bg_loss = -torch.log(assignment.background_prob[bg_mask].clamp_min(1e-6)).mean()
        return fg_loss, bg_loss

    def _loss_occ_local2gs(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        bridges = outputs.get("bridges")
        if bridges is None:
            return _zero_like(outputs["dynamic"].dyn_logit_1_4)
        target = torch.cat([outputs["entities"].refined_instance_affinity, outputs["entities"].refined_motion_code], dim=-1).detach()
        return F.smooth_l1_loss(bridges.occ_to_gs_local, target)

    def _loss_gs2occ_local(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        bridges = outputs.get("bridges")
        if bridges is None:
            return _zero_like(outputs["dynamic"].dyn_logit_1_4)
        target = outputs["occ"].occ_volumes["occ_1"].detach()
        return F.mse_loss(bridges.gs_to_occ_local, target)

    def _loss_sem_proj_2d(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "render" not in outputs or "sam3_semantic_full" not in batch:
            return _zero_like(outputs["dynamic"].dyn_logit_1_4)
        pred = outputs["render"].sem_proj_2d
        target = batch["sam3_semantic_full"]
        return _balanced_bce_with_logits(pred, target) + _soft_dice_loss_from_logits(pred, target)

    def _loss_motion_cons(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        gauss_motion = outputs["gaussians"].motion_code[..., : outputs["entities"].refined_motion_code.shape[-1]]
        return F.smooth_l1_loss(outputs["entities"].refined_motion_code, gauss_motion)

    def _loss_gs_sparse(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        opacity = outputs["gaussians"].opacity
        return opacity.mean()

    def _build_query_targets_hungarian(
        self,
        batch: Dict[str, torch.Tensor],
        query_feat: torch.Tensor,
        query_box3d: torch.Tensor,
        query_motion: torch.Tensor,
        query_cls_logit: torch.Tensor,
        box_source: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        track_boxes = batch["track_boxes_3d"]
        track_cls = batch["track_cls"]
        track_visible = batch["track_visible"]
        target_source = track_boxes if box_source is None else box_source

        b, source_t, _, target_dim = target_source.shape
        total_queries = query_box3d.shape[2]
        matched_gt_slot = track_cls.new_full((b, source_t, total_queries), -1)
        cls_target = track_cls.new_full((b, source_t, total_queries), -1)
        box_target = target_source.new_zeros((b, source_t, total_queries, target_dim))
        visible_mask = torch.zeros((b, source_t, total_queries), dtype=torch.bool, device=track_boxes.device)
        match_cost = target_source.new_zeros((b, source_t, total_queries))

        for batch_idx in range(b):
            prev_feat = None
            prev_id = None
            prev_active = None
            for time_idx in range(source_t):
                delta_t = batch["track_delta_box"][batch_idx, time_idx] if time_idx < batch["track_delta_box"].shape[1] else None
                match = match_queries_to_tracks(
                    query_feat=query_feat[batch_idx, time_idx],
                    query_box3d=query_box3d[batch_idx, time_idx],
                    query_motion=query_motion[batch_idx, time_idx],
                    query_cls_logit=query_cls_logit[batch_idx, time_idx],
                    track_boxes_t=track_boxes[batch_idx, time_idx],
                    track_cls_t=track_cls[batch_idx, time_idx],
                    track_visible_t=track_visible[batch_idx, time_idx],
                    track_id_t=batch["track_id"][batch_idx, time_idx],
                    prev_query_feat=prev_feat,
                    prev_query_id=prev_id,
                    prev_active_mask=prev_active,
                    track_delta_t=delta_t,
                )
                matched_queries = torch.nonzero(match.query_to_gt >= 0, as_tuple=False).squeeze(-1)
                if matched_queries.numel() > 0:
                    gt_slots = match.query_to_gt[matched_queries]
                    matched_gt_slot[batch_idx, time_idx, matched_queries] = gt_slots
                    cls_target[batch_idx, time_idx, matched_queries] = track_cls[batch_idx, time_idx, gt_slots]
                    box_target[batch_idx, time_idx, matched_queries] = target_source[batch_idx, time_idx, gt_slots]
                    visible_mask[batch_idx, time_idx, matched_queries] = True
                    match_cost[batch_idx, time_idx, matched_queries] = match.query_match_cost[matched_queries].to(
                        match_cost.dtype
                    )

                prev_feat = query_feat[batch_idx, time_idx].detach()
                prev_id = match.query_to_track_id.detach()
                prev_active = match.query_to_gt >= 0
        return matched_gt_slot, cls_target, box_target, visible_mask, match_cost

    def _build_gaussian_query_targets(self, batch: Dict[str, torch.Tensor], max_tracks: int) -> torch.Tensor:
        global_ids = batch["gs_global_track_id_1_8"]
        track_ids = batch["track_id"]
        track_visible = batch["track_visible"]
        b, t, v, h, w = global_ids.shape
        _, _, n = track_ids.shape
        target_slot = torch.full((b, t * v * h * w), -1, dtype=torch.long, device=global_ids.device)

        for batch_idx in range(b):
            slot_track_ids = {}
            for slot_idx in range(min(n, max_tracks)):
                visible_steps = torch.nonzero(track_visible[batch_idx, :, slot_idx], as_tuple=False).squeeze(-1)
                if visible_steps.numel() == 0:
                    continue
                slot_track_ids[int(track_ids[batch_idx, int(visible_steps[0]), slot_idx])] = slot_idx

            if not slot_track_ids:
                continue

            flat_ids = global_ids[batch_idx].reshape(-1)
            for track_id, slot_idx in slot_track_ids.items():
                target_slot[batch_idx][flat_ids == track_id] = int(slot_idx)
        return target_slot
