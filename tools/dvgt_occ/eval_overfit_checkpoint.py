#!/usr/bin/env python3
"""Evaluate a stage-B checkpoint on a tiny cached clip subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dvgt_occ.data import DVGTOccClipDataset
from dvgt_occ.models import DVGTOccModel
from dvgt_occ.training import (
    DEFAULT_CACHE_KEYS,
    DEFAULT_SUPERVISION_KEYS,
    DVGTOccLossBuilder,
    collate_dvgt_occ_batch,
    move_batch_to_device,
)
from train_overfit import build_loss_weights, build_model_config, build_runtime_config, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a stage-B overfit checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _binary_iou_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits > 0
    truth = target > 0.5
    inter = torch.logical_and(pred, truth).sum().item()
    union = torch.logical_or(pred, truth).sum().item()
    return float(inter / union) if union > 0 else 1.0


def _mean_prob_on_masks(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    prob = torch.sigmoid(logits)
    pos_mask = target > 0.5
    neg_mask = ~pos_mask
    pos_mean = float(prob[pos_mask].mean().item()) if torch.any(pos_mask) else 0.0
    neg_mean = float(prob[neg_mask].mean().item()) if torch.any(neg_mask) else 0.0
    return pos_mean, neg_mean


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = build_runtime_config(
        argparse.Namespace(
            manifest=None,
            output_root=None,
            limit=None,
            scene_ids=None,
            clip_ids=None,
            max_steps=None,
            batch_size=None,
            num_workers=None,
            seed=None,
            device=None,
            log_dir=None,
        ),
        cfg,
    )
    model_cfg = build_model_config(cfg)
    device = torch.device(args.device)

    dataset = DVGTOccClipDataset(
        manifest_path=runtime["manifest"],
        root=runtime["output_root"],
        load_cache=True,
        load_supervision=True,
        projected_semantic_classes=model_cfg.projected_semantic_classes,
        cache_keys=DEFAULT_CACHE_KEYS,
        supervision_keys=DEFAULT_SUPERVISION_KEYS,
        scene_ids=runtime["scene_ids"],
        clip_ids=runtime["clip_ids"],
        limit=runtime["limit"],
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_dvgt_occ_batch)

    model = DVGTOccModel(config=model_cfg).to(device)
    loss_builder = DVGTOccLossBuilder(config=model_cfg, weights=build_loss_weights(cfg)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    records = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss_total, loss_dict = loss_builder(outputs, batch)

            dyn_logit = outputs["dynamic"].dyn_logit_1_4.squeeze(3)
            dyn_target = batch["dyn_soft_mask_1_4"]
            dyn_iou_1_4 = _binary_iou_from_logits(dyn_logit, dyn_target)
            dyn_pos_prob, dyn_neg_prob = _mean_prob_on_masks(dyn_logit, dyn_target)

            dyn_full = outputs["dynamic"].dyn_logit_full.squeeze(3)
            target_full = torch.nn.functional.interpolate(
                dyn_target.reshape(-1, 1, dyn_target.shape[-2], dyn_target.shape[-1]),
                size=dyn_full.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape_as(dyn_full)
            dyn_iou_full = _binary_iou_from_logits(dyn_full, target_full)

            occ_logits = outputs["occ"].occ_logit.squeeze(2)
            occ_iou = _binary_iou_from_logits(occ_logits, batch["occ_label"])
            occ_pos_prob, occ_neg_prob = _mean_prob_on_masks(occ_logits, batch["occ_label"])

            visible_tracks = int(batch["track_visible"].any(dim=1).sum().item())
            presence_logits = outputs["queries"].presence_logit[0]
            presence_above_zero = int((presence_logits > 0).sum().item())
            topk = int(max(visible_tracks, 1))
            presence_topk_mean = float(presence_logits.topk(topk).values.mean().item())
            presence_rest_mean = float(
                presence_logits.topk(max(presence_logits.numel() - topk, 1), largest=False).values.mean().item()
            )

            record = {
                "scene_id": batch["scene_id"][0],
                "clip_id": batch["clip_id"][0],
                "loss_total": float(loss_total.item()),
                "dyn_iou_1_4": dyn_iou_1_4,
                "dyn_iou_full_proxy": dyn_iou_full,
                "dyn_pos_prob_mean": dyn_pos_prob,
                "dyn_neg_prob_mean": dyn_neg_prob,
                "occ_iou": occ_iou,
                "occ_pos_prob_mean": occ_pos_prob,
                "occ_neg_prob_mean": occ_neg_prob,
                "visible_tracks": visible_tracks,
                "presence_queries_gt0": presence_above_zero,
                "presence_topk_mean": presence_topk_mean,
                "presence_rest_mean": presence_rest_mean,
            }
            for key, value in loss_dict.items():
                record[f"loss_{key}"] = float(value.item())
            records.append(record)

    print(json.dumps({"checkpoint": str(args.checkpoint), "results": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
