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

INVALID_MATCH_COST = 1e5


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


def _masked_balanced_bce_with_logits(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = (valid > 0.5).float()
    if valid.shape != pred.shape:
        valid = valid.expand_as(pred)
    if float(valid.sum().item()) <= 0:
        return _zero_like(pred)
    pred_valid = pred[valid > 0.5]
    target_valid = target.float()[valid > 0.5]
    return _balanced_bce_with_logits(pred_valid, target_valid)


def _masked_soft_dice_loss_from_logits(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    valid = (valid > 0.5).float()
    if valid.shape != pred.shape:
        valid = valid.expand_as(pred)
    if float(valid.sum().item()) <= 0:
        return _zero_like(pred)
    prob = torch.sigmoid(pred) * valid
    target = target.float() * valid
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
        self._lpips = None
        self._lpips_available = False
        try:
            import lpips  # type: ignore

            self._lpips = lpips.LPIPS(net="alex")
            self._lpips.requires_grad_(False)
            self._lpips_available = True
        except Exception:
            self._lpips = None
            self._lpips_available = False

    def forward(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reference = outputs["dynamic"].dyn_logit_1_4
        losses: Dict[str, torch.Tensor] = {
            key: _zero_like(reference) for key in asdict(self.weights).keys()
        }

        if self.weights.dyn_soft > 0.0:
            losses["dyn_soft"] = self._loss_dyn_soft(outputs, batch)
        if any(
            weight > 0.0
            for weight in (
                self.weights.presence,
                self.weights.query_match,
                self.weights.query_cls,
                self.weights.query_box3d,
            )
        ) and outputs.get("queries") is not None:
            (
                losses["presence"],
                losses["query_match"],
                losses["query_cls"],
                losses["query_box3d"],
            ) = self._loss_queries(outputs, batch)
        if self.weights.track_mem_feat > 0.0 and outputs.get("entities") is not None:
            losses["track_mem_feat"] = self._loss_entity_motion(outputs, batch)
        if self.weights.track_mem_box > 0.0 and outputs.get("queries") is not None:
            losses["track_mem_box"] = self._loss_query_motion(outputs, batch)
        if self.weights.birth_death > 0.0 and outputs.get("queries") is not None:
            losses["birth_death"] = self._loss_birth_death(outputs, batch)
        if self.weights.occ > 0.0 and outputs.get("occ") is not None:
            losses["occ"] = self._loss_occ(outputs, batch)
        if self.weights.sem_occ > 0.0 and outputs.get("occ") is not None:
            losses["sem_occ"] = self._loss_sem_occ(outputs, batch)
        if self.weights.dyn_occ > 0.0 and outputs.get("occ") is not None:
            losses["dyn_occ"] = self._loss_dyn_occ(outputs, batch)
        if self.weights.gs_render > 0.0 and outputs.get("render") is not None:
            losses["gs_render"] = self._loss_gs_render(outputs, batch)
        if self.weights.sky_alpha > 0.0 and outputs.get("render") is not None:
            losses["sky_alpha"] = self._loss_sky_alpha(outputs, batch)
        if self.weights.inst_contrast > 0.0 and outputs.get("entities") is not None:
            losses["inst_contrast"] = self._loss_instance_contrast(outputs, batch)
        if any(weight > 0.0 for weight in (self.weights.query2gs, self.weights.q2gs_null)) and outputs.get("assignment") is not None:
            losses["query2gs"], losses["q2gs_null"] = self._loss_query_to_gaussian(outputs, batch)
        if self.weights.mask_render > 0.0 and outputs.get("render") is not None:
            losses["mask_render"] = self._loss_dyn_mask_full(outputs, batch)
        if self.weights.occ_local2gs > 0.0 and outputs.get("bridges") is not None and outputs.get("entities") is not None:
            losses["occ_local2gs"] = self._loss_occ_local2gs(outputs, batch)
        if self.weights.gs2occ_local > 0.0 and outputs.get("bridges") is not None:
            losses["gs2occ_local"] = self._loss_gs2occ_local(outputs, batch)
        if self.weights.sem_proj_2d > 0.0 and outputs.get("render") is not None:
            losses["sem_proj_2d"] = self._loss_sem_proj_2d(outputs, batch)
        if self.weights.motion_cons > 0.0 and outputs.get("entities") is not None and outputs.get("gaussians") is not None:
            losses["motion_cons"] = self._loss_motion_cons(outputs, batch)
        if self.weights.gs_sparse > 0.0 and outputs.get("gaussians") is not None:
            losses["gs_sparse"] = self._loss_gs_sparse(outputs, batch)
        losses["ddp_tether"] = self._loss_ddp_tether(outputs)

        total = _zero_like(reference)
        for key, weight in asdict(self.weights).items():
            total = total + float(weight) * losses[key]
        total = total + losses["ddp_tether"]
        return total, losses

    def _loss_dyn_soft(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = outputs["dynamic"].dyn_logit_1_4.squeeze(3)
        target = batch["dyn_soft_mask_1_4"]
        return _balanced_bce_with_logits(pred, target) + _soft_dice_loss_from_logits(pred, target)

    def _loss_dyn_mask_full(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        train_mode = str(batch.get("_train_mode", "v1-stable"))
        if train_mode == "v1-gs-aux" and "sam3_dyn_mask_full" in batch and outputs.get("dynamic") is not None:
            pred_dyn = outputs["dynamic"].dyn_logit_1_4.squeeze(3)
            target_dyn = batch["sam3_dyn_mask_full"]
            valid = batch.get("sam3_valid_mask_full")
            target_is_source = batch.get("target_is_source")
            if target_is_source is not None and target_dyn.shape[2] != pred_dyn.shape[2]:
                target_src = target_dyn.new_zeros((target_dyn.shape[0], target_dyn.shape[1], pred_dyn.shape[2], target_dyn.shape[3], target_dyn.shape[4]))
                valid_src = None if valid is None else valid.new_zeros((valid.shape[0], valid.shape[1], pred_dyn.shape[2], valid.shape[3], valid.shape[4]))
                for batch_idx in range(target_dyn.shape[0]):
                    src_indices = torch.nonzero(target_is_source[batch_idx], as_tuple=False).flatten()[: pred_dyn.shape[2]]
                    if src_indices.numel() == 0:
                        continue
                    take = int(src_indices.numel())
                    target_src[batch_idx, :, :take] = target_dyn[batch_idx, :, src_indices]
                    if valid_src is not None and valid is not None:
                        valid_src[batch_idx, :, :take] = valid[batch_idx, :, src_indices]
                target_dyn = target_src
                valid = valid_src
            b, t, v, h, w = target_dyn.shape
            target_dyn = F.interpolate(
                target_dyn.reshape(b * t * v, 1, h, w),
                size=pred_dyn.shape[-2:],
                mode="nearest",
            ).reshape(b, t, v, pred_dyn.shape[-2], pred_dyn.shape[-1])
            if valid is not None:
                valid = F.interpolate(
                    valid.reshape(b * t * v, 1, h, w).float(),
                    size=pred_dyn.shape[-2:],
                    mode="nearest",
                ).reshape(b, t, v, pred_dyn.shape[-2], pred_dyn.shape[-1])
                return _masked_balanced_bce_with_logits(pred_dyn, target_dyn, valid) + _masked_soft_dice_loss_from_logits(pred_dyn, target_dyn, valid)
            return _balanced_bce_with_logits(pred_dyn, target_dyn) + _soft_dice_loss_from_logits(pred_dyn, target_dyn)

        if "sam3_dyn_mask_full" in batch and "render" in outputs:
            mask_all_weight = float(batch.get("_mask_all_weight", 0.0))
            pred_dyn = _safe_logit(outputs["render"].render_alpha_dynamic.squeeze(3))
            target_dyn = batch["sam3_dyn_mask_full"]
            valid = batch.get("sam3_valid_mask_full")
            if valid is not None:
                loss_dyn = _masked_balanced_bce_with_logits(pred_dyn, target_dyn, valid) + _masked_soft_dice_loss_from_logits(pred_dyn, target_dyn, valid)
            else:
                loss_dyn = _balanced_bce_with_logits(pred_dyn, target_dyn) + _soft_dice_loss_from_logits(pred_dyn, target_dyn)
            if train_mode == "v1-stable" or mask_all_weight <= 0.0:
                return loss_dyn
            pred_all = _safe_logit(outputs["render"].render_alpha_all.squeeze(3))
            target_all = batch.get("sam3_all_mask_full", target_dyn)
            if valid is not None:
                loss_all = _masked_balanced_bce_with_logits(pred_all, target_all, valid) + _masked_soft_dice_loss_from_logits(pred_all, target_all, valid)
            else:
                loss_all = _balanced_bce_with_logits(pred_all, target_all) + _soft_dice_loss_from_logits(pred_all, target_all)
            return loss_dyn + mask_all_weight * loss_all

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

    def _loss_gs_render(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "render" not in outputs or "rgb_target" not in batch:
            return _zero_like(outputs["dynamic"].dyn_logit_1_4)

        pred = outputs["render"].render_rgb_all
        if outputs["render"].render_rgb_all_composited is not None:
            pred = outputs["render"].render_rgb_all_composited
        target = batch["rgb_target"]

        valid = torch.ones_like(pred[:, :, :, :1])
        target_is_source = batch.get("target_is_source")
        if target_is_source is not None:
            source_weight = float(getattr(self.config, "render_source_weight", 1.0))
            heldout_weight = float(getattr(self.config, "render_heldout_weight", 0.1))
            source_mask = target_is_source[:, None, :, None, None, None].to(dtype=valid.dtype, device=valid.device)
            view_weight = source_mask * source_weight + (1.0 - source_mask) * heldout_weight
            valid = valid * view_weight

        valid_sum = valid.sum().clamp_min(1.0)
        l1 = ((pred - target).abs() * valid).sum() / valid_sum

        if self._lpips_available and self._lpips is not None:
            lpips_loss = self._masked_lpips(pred, target, valid)
        else:
            lpips_loss = self._multiscale_l1(pred, target, valid)
        lpips_base_weight = float(getattr(self.config, "render_lpips_weight", 0.05))
        lpips_weight = lpips_base_weight * min(float(self._current_step_hint(outputs, batch)) / 1000.0, 1.0)
        gs_conf_loss = _zero_like(l1)
        scale_reg = _zero_like(l1)
        color_sup = _zero_like(l1)
        gaussians = outputs.get("gaussians")
        if gaussians is not None:
            scale_reg = torch.mean(torch.relu(gaussians.scale - 0.30))
            feat_dc = gaussians.feat_dc
            b_g, t_g, v_g, h_g, w_g, _ = feat_dc.shape
            gt_rgb = batch.get("source_rgb", batch.get("rgb_source", batch["rgb_target"]))
            if gt_rgb.shape[:3] == feat_dc.shape[:3]:
                gt_down = torch.nn.functional.interpolate(
                    gt_rgb.reshape(b_g * t_g * v_g, 3, gt_rgb.shape[-2], gt_rgb.shape[-1]),
                    size=(h_g, w_g), mode="bilinear", align_corners=False,
                ).reshape(b_g, t_g, v_g, 3, h_g, w_g).permute(0, 1, 2, 4, 5, 3)
                color_sup = (feat_dc - gt_down).abs().mean()
        return l1 + lpips_weight * lpips_loss + 0.01 * gs_conf_loss + 0.1 * scale_reg + 1.0 * color_sup

    def _loss_sky_alpha(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        sky_mask = batch.get("sky_mask_full")
        if sky_mask is None:
            return _zero_like(outputs["dynamic"].dyn_logit_1_4)
        pred = outputs["render"].render_alpha_all.squeeze(3)
        sky = sky_mask.float()
        denom = sky.sum().clamp_min(1.0)
        return (pred * sky).sum() / denom

    def _loss_occ(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = outputs["occ"].occ_logit.squeeze(2)
        target = batch["occ_label"]
        return _balanced_bce_with_logits(pred, target) + _soft_dice_loss_from_logits(pred, target)

    def _loss_sem_occ(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = outputs["occ"].sem_logit
        b, t, c, z, y, x = pred.shape
        target = batch["occ_sem_label"]
        return F.cross_entropy(pred.reshape(b * t, c, z, y, x), target.reshape(b * t, z, y, x))

    def _loss_dyn_occ(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = outputs["occ"].dyn_occ_logit.squeeze(2)
        target = batch["occ_dyn_label"]
        return _balanced_bce_with_logits(pred, target) + _soft_dice_loss_from_logits(pred, target)

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
            fg_bank = assignment.assignment_prob[..., 1:]
            if fg_bank.shape[-1] == 0:
                return fg_loss, _zero_like(assignment.assignment_prob)
            safe_slot = target_slot.clamp(min=0)
            valid_slot = safe_slot < fg_bank.shape[-1]
            active_fg = fg_mask & valid_slot
            if torch.any(active_fg):
                fg_prob = fg_bank.gather(-1, safe_slot.clamp(max=fg_bank.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
                fg_loss = -torch.log(fg_prob[active_fg].clamp_min(1e-6)).mean()

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
        valid = batch.get("sam3_valid_mask_full")
        if valid is None:
            return _balanced_bce_with_logits(pred, target) + _soft_dice_loss_from_logits(pred, target)
        valid = valid.unsqueeze(3)
        return _masked_balanced_bce_with_logits(pred, target, valid) + _masked_soft_dice_loss_from_logits(pred, target, valid)

    def _loss_motion_cons(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if outputs.get("entities") is None:
            return _zero_like(outputs["dynamic"].dyn_logit_1_4)
        gauss_motion = outputs["gaussians"].motion_code[..., : outputs["entities"].refined_motion_code.shape[-1]]
        return F.smooth_l1_loss(outputs["entities"].refined_motion_code, gauss_motion)

    def _loss_gs_sparse(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        opacity = outputs["gaussians"].opacity
        return opacity.mean()

    def _build_render_valid_mask(self, points_conf: torch.Tensor, threshold: float = 0.30, dilate_px: int = 5) -> torch.Tensor:
        b, t, v, h, w = points_conf.shape
        conf = points_conf.float().reshape(b * t * v, h * w)

        # DGGT-style point/depth confidence is not guaranteed to be a 0..1
        # probability map. In practice our cached points_conf often comes from an
        # expp1-style head and lives on a >1 scale, so a raw `> 0.30` threshold
        # collapses into an all-ones valid mask. Normalize each frame-view
        # robustly before applying the stable-contract threshold.
        q05 = torch.quantile(conf, 0.05, dim=1, keepdim=True)
        q95 = torch.quantile(conf, 0.95, dim=1, keepdim=True)
        denom = (q95 - q05).clamp_min(1e-6)
        conf_norm = ((conf - q05) / denom).clamp(0.0, 1.0)

        # If a map is already probability-like and spread is tiny, keep it
        # directly instead of turning the whole frame valid.
        raw_min = conf.min(dim=1, keepdim=True).values
        raw_max = conf.max(dim=1, keepdim=True).values
        looks_prob = (raw_min >= 0.0) & (raw_max <= 1.0 + 1e-4)
        conf_for_mask = torch.where(looks_prob, conf.clamp(0.0, 1.0), conf_norm)

        mask = (conf_for_mask > threshold).float().reshape(b * t * v, 1, h, w)
        if dilate_px > 0:
            kernel = dilate_px * 2 + 1
            mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilate_px)
        return mask.reshape(b, t, v, 1, h, w)

    def _masked_lpips(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, t, v, c, h, w = pred.shape
        pred_m = (pred * valid).reshape(b * t * v, c, h, w)
        tgt_m = (target * valid).reshape(b * t * v, c, h, w)
        lpips_module = self._lpips.to(pred.device)
        value = lpips_module(pred_m * 2.0 - 1.0, tgt_m * 2.0 - 1.0)
        return value.mean()

    def _multiscale_l1(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        loss = _zero_like(pred)
        pred_s = pred
        target_s = target
        valid_s = valid
        for _ in range(3):
            denom = valid_s.sum().clamp_min(1.0)
            loss = loss + ((pred_s - target_s).abs() * valid_s).sum() / denom
            if pred_s.shape[-1] <= 28 or pred_s.shape[-2] <= 14:
                break
            pred_s = F.avg_pool2d(pred_s.reshape(-1, pred_s.shape[3], pred_s.shape[4], pred_s.shape[5]), 2, 2).reshape(
                pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3], pred_s.shape[-2] // 2, pred_s.shape[-1] // 2
            )
            target_s = F.avg_pool2d(
                target_s.reshape(-1, target_s.shape[3], target_s.shape[4], target_s.shape[5]), 2, 2
            ).reshape(pred_s.shape)
            valid_s = F.avg_pool2d(valid_s.reshape(-1, 1, valid_s.shape[-2], valid_s.shape[-1]), 2, 2).reshape(
                pred_s.shape[0], pred_s.shape[1], pred_s.shape[2], 1, pred_s.shape[-2], pred_s.shape[-1]
            )
            valid_s = (valid_s > 0.25).float()
        return loss / 3.0

    def _current_step_hint(self, outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> int:
        step_hint = outputs.get("step_hint")
        if isinstance(step_hint, torch.Tensor):
            return int(step_hint.item())
        if isinstance(step_hint, int):
            return int(step_hint)
        return 0

    def _loss_ddp_tether(self, outputs: Dict[str, object]) -> torch.Tensor:
        reference = outputs["dynamic"].dyn_logit_1_4
        tether = _zero_like(reference)
        tether = tether + outputs["dynamic"].dyn_logit_full.sum() * 0.0
        queries = outputs.get("queries")
        if queries is not None:
            tether = tether + queries.query_cls_logit.sum() * 0.0
            tether = tether + queries.query_motion.sum() * 0.0
        occ = outputs.get("occ")
        occ_aux = getattr(occ, "aux_decoder_full", None)
        if occ_aux is not None:
            tether = tether + occ_aux.sum() * 0.0
        gaussians = outputs.get("gaussians_pre_merge")
        if gaussians is None:
            gaussians = outputs.get("gaussians")
        gs_aux = getattr(gaussians, "aux_decoder_full", None)
        if gs_aux is not None:
            tether = tether + gs_aux.sum() * 0.0
        gs_dynamic_logit = getattr(gaussians, "dynamic_logit", None)
        if gs_dynamic_logit is not None:
            tether = tether + gs_dynamic_logit.sum() * 0.0
        bridges = outputs.get("bridges")
        if bridges is not None:
            tether = tether + bridges.gs_to_occ_local.sum() * 0.0
        render = outputs.get("render")
        if render is not None:
            tether = tether + render.sem_proj_2d.sum() * 0.0
        return tether

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
                    valid_queries = matched_queries[match.query_match_cost[matched_queries] < INVALID_MATCH_COST]
                    if valid_queries.numel() > 0:
                        gt_slots = match.query_to_gt[valid_queries]
                        matched_gt_slot[batch_idx, time_idx, valid_queries] = gt_slots
                        cls_target[batch_idx, time_idx, valid_queries] = track_cls[batch_idx, time_idx, gt_slots]
                        box_target[batch_idx, time_idx, valid_queries] = target_source[batch_idx, time_idx, gt_slots]
                        visible_mask[batch_idx, time_idx, valid_queries] = True
                        match_cost[batch_idx, time_idx, valid_queries] = match.query_match_cost[valid_queries].to(
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
