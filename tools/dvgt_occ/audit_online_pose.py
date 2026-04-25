#!/usr/bin/env python3
"""Audit online DVGT point/pose coordinates against GT ego diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tools.dvgt_occ.train import (
    build_dataset,
    build_online_dvgt_frontend,
    build_runtime_config,
    load_config,
    move_batch_to_device,
    set_seed,
)
from dvgt_occ import DVGTOccConfig
from dvgt_occ.training import collate_dvgt_occ_batch
from dvgt.evaluation.utils.geometry import accumulate_transform_points_and_pose_to_first_frame
from dvgt.utils.pose_encoding import decode_pose


def _rotation_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rel = a[..., :3, :3].transpose(-1, -2) @ b[..., :3, :3]
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def _accumulate_without_step_inverse(t_step: torch.Tensor) -> torch.Tensor:
    b, t = t_step.shape[:2]
    out = torch.eye(4, device=t_step.device, dtype=t_step.dtype).view(1, 1, 4, 4).repeat(b, t, 1, 1)
    cur = out[:, 0]
    for idx in range(1, t):
        cur = cur @ t_step[:, idx]
        out[:, idx] = cur
    return out


def _pose_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    trans = (pred[..., :3, 3] - gt[..., :3, 3]).norm(dim=-1)
    rot = _rotation_angle_deg(pred, gt)
    trans_no0 = trans[:, 1:] if trans.shape[1] > 1 else trans
    rot_no0 = rot[:, 1:] if rot.shape[1] > 1 else rot
    return {
        "trans_mean": float(trans_no0.mean().item()),
        "trans_max": float(trans_no0.max().item()),
        "rot_deg_mean": float(rot_no0.mean().item()),
        "rot_deg_max": float(rot_no0.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dvgt_occ/train_stage_c_gs_only_v57_online_dvgt_varvt_posepred.yaml"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("data/nuscenes_dvgt_v0/logs/pose_audit_v57/pose_audit.jsonl"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = build_runtime_config(argparse.Namespace(
        config=args.config,
        manifest=None,
        output_root=None,
        train_scene_ids=None,
        val_scene_ids=None,
        train_limit=args.samples,
        val_limit=None,
        max_steps=None,
        batch_size=1,
        num_workers=0,
        seed=57,
        device=args.device,
        log_dir=None,
        log_interval=None,
        save_interval=None,
        val_interval=None,
        val_batches=None,
        resume=None,
        no_resume=True,
        ddp_timeout_sec=None,
        ddp_backend=None,
        find_unused_parameters=False,
        broadcast_buffers=False,
    ), cfg)
    set_seed(int(runtime["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_cfg = replace(DVGTOccConfig(), **dict(cfg.get("model", {})))
    dataset = build_dataset(runtime["manifest"], runtime["output_root"], model_cfg, runtime["train_scene_ids"], args.samples, runtime)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_dvgt_occ_batch)
    frontend = build_online_dvgt_frontend(runtime, model_cfg, device)
    frontend.eval()

    try:
        import diffusers  # type: ignore
        diffusers_available = True
        diffusers_version = getattr(diffusers, "__version__", "unknown")
    except Exception:
        diffusers_available = False
        diffusers_version = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                out = frontend(batch["source_rgb"])
                rel = out["relative_ego_pose_enc"].float()
                points = out["points"].float()
                conf = out["points_conf"].float()
                decoded, _ = decode_pose(rel, pose_encoding_type="absT_quaR")
                decoded = decoded.float()

                gt = batch["gt_ego_n_to_first_ego"].float()
                variants = {}
                for pose_scale in (1.0, float(runtime["online_points_metric_scale"])):
                    step = decoded.clone()
                    step[..., :3, 3] *= pose_scale
                    pred_accum, _ = accumulate_transform_points_and_pose_to_first_frame(step, points)
                    variants[f"accumulate_step_inverse_pose_scale_{pose_scale:g}"] = _pose_metrics(pred_accum, gt)
                    pred_noinv = _accumulate_without_step_inverse(step)
                    variants[f"accumulate_no_step_inverse_pose_scale_{pose_scale:g}"] = _pose_metrics(pred_noinv, gt)

                record = {
                    "sample_idx": idx,
                    "scene_id": batch["scene_id"],
                    "clip_id": batch["clip_id"],
                    "source_view_ids": batch["source_view_ids"],
                    "target_view_ids": batch["target_view_ids"],
                    "frame_ids_10hz": batch["frame_ids_10hz"],
                    "online_sample_meta": batch.get("online_sample_meta"),
                    "diffusers_available": diffusers_available,
                    "diffusers_version": diffusers_version,
                    "rel_pose_translation_abs_mean": float(rel[..., :3].abs().mean().item()),
                    "rel_pose_quat_norm_mean": float(rel[..., 3:7].norm(dim=-1).mean().item()),
                    "points_abs_mean_raw": float(points.abs().mean().item()),
                    "points_conf_mean": float(conf.mean().item()),
                    "gt_trans_norm_mean": float(gt[..., :3, 3].norm(dim=-1).mean().item()),
                    "variants": variants,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
