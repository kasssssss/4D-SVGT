#!/usr/bin/env python3
"""Audit the original DVGT2 pose decoder and post-processing path."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dvgt_occ.train import (  # noqa: E402
    build_dataset,
    build_runtime_config,
    load_config,
    move_batch_to_device,
    set_seed,
)
from dvgt_occ import DVGTOccConfig  # noqa: E402
from dvgt_occ.training import collate_dvgt_occ_batch  # noqa: E402
from dvgt.models.architectures.dvgt2 import DVGT2  # noqa: E402
from dvgt.utils.pose_encoding import decode_pose  # noqa: E402


def _rotation_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rel = a[..., :3, :3].transpose(-1, -2) @ b[..., :3, :3]
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def _pose_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, object]:
    trans = (pred[..., :3, 3] - gt[..., :3, 3]).norm(dim=-1)
    rot = _rotation_angle_deg(pred, gt)
    trans_no0 = trans[:, 1:] if trans.shape[1] > 1 else trans
    rot_no0 = rot[:, 1:] if rot.shape[1] > 1 else rot
    return {
        "trans_mean": float(trans_no0.mean().item()),
        "trans_max": float(trans_no0.max().item()),
        "rot_deg_mean": float(rot_no0.mean().item()),
        "rot_deg_max": float(rot_no0.max().item()),
        "trans_per_frame": [float(x) for x in trans[0].detach().cpu()],
        "rot_deg_per_frame": [float(x) for x in rot[0].detach().cpu()],
    }


def _compute_ego_past_to_ego_curr(t_first_to_n: torch.Tensor) -> torch.Tensor:
    """Torch port of dvgt.utils.geometry.compute_ego_past_to_ego_curr."""
    b, t = t_first_to_n.shape[:2]
    out = torch.eye(4, device=t_first_to_n.device, dtype=t_first_to_n.dtype).view(1, 1, 4, 4).repeat(b, t, 1, 1)
    if t <= 1:
        return out
    t_first_curr = t_first_to_n[:, 1:]
    t_curr_first = torch.linalg.inv(t_first_curr)
    t_first_past = t_first_to_n[:, :-1]
    out[:, 1:] = t_curr_first @ t_first_past
    return out


def _make_original_dvgt(checkpoint_path: Path, dino_path: str | None, device: torch.device) -> tuple[DVGT2, dict[str, object]]:
    dvgt = DVGT2(
        dino_v3_weight_path=dino_path,
        use_causal_mask=True,
        future_frame_window=8,
        relative_pose_window=1,
        ego_pose_head_conf={
            "_target_": "dvgt.models.heads.dvgt2_ego_pose_head.DVGT2EgoPoseHead",
            "max_frames": 48,
            "future_frame_window": 8,
            "relative_pose_window": 1,
        },
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    has_pose_head = any(str(key).startswith("ego_pose_head.") for key in state.keys())
    missing, unexpected = dvgt.load_state_dict(state, strict=False)
    dvgt.to(device).eval()
    for parameter in dvgt.parameters():
        parameter.requires_grad_(False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "has_pose_head": has_pose_head,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_sample": list(missing)[:16],
        "unexpected_sample": list(unexpected)[:16],
    }
    return dvgt, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dvgt_occ/train_stage_c_gs_only_v58_online_dvgt_varvt_localego.yaml"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("data/nuscenes_dvgt_v0/logs/original_dvgt_pose_audit_v1/pose_audit.jsonl"))
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-views", type=int, default=None)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--temporal-strides", type=int, nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = build_runtime_config(
        argparse.Namespace(
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
            seed=58,
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
        ),
        cfg,
    )
    if args.min_frames is not None:
        runtime["online_min_frames"] = int(args.min_frames)
    if args.max_frames is not None:
        runtime["online_max_frames"] = int(args.max_frames)
    if args.min_views is not None:
        runtime["online_min_views"] = int(args.min_views)
    if args.max_views is not None:
        runtime["online_max_views"] = int(args.max_views)
    if args.temporal_strides:
        runtime["online_temporal_strides"] = tuple(int(x) for x in args.temporal_strides)
    set_seed(int(runtime["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_cfg = replace(DVGTOccConfig(), **dict(cfg.get("model", {})))
    dataset = build_dataset(runtime["manifest"], runtime["output_root"], model_cfg, runtime["train_scene_ids"], args.samples, runtime)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_dvgt_occ_batch)

    checkpoint_path = Path(runtime["online_checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    dino_path = runtime.get("online_dino_v3_weight_path")
    dvgt, load_summary = _make_original_dvgt(checkpoint_path, str(dino_path) if dino_path else None, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for sample_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            images = batch["source_rgb"]
            gt_metric = batch["gt_ego_n_to_first_ego"].float()
            scale = float(runtime.get("online_points_metric_scale", 10.0))
            gt_scale_factor = 1.0 / scale
            gt_scaled = gt_metric.clone()
            gt_scaled[..., :3, 3] *= gt_scale_factor
            ego_past_scaled = _compute_ego_past_to_ego_curr(gt_scaled)

            with torch.no_grad():
                predictions = dvgt(images)
                eval_batch = {
                    "ego_past_to_ego_curr": ego_past_scaled.clone(),
                    "points_in_ego_n": predictions["points"].detach().clone(),
                    "ray_depth": predictions["points"].detach().norm(dim=-1).clone(),
                }
                outputs = dvgt.post_process_for_eval(
                    predictions,
                    eval_batch,
                    eval_traj=False,
                    eval_point_and_pose=True,
                    gt_scale_factor=gt_scale_factor,
                )
                rel_pose_enc = dvgt.ego_pose_head.post_process_pose_for_eval(predictions)
                rel_pose_decoded, _ = decode_pose(rel_pose_enc.float(), pose_encoding_type="absT_quaR")

            pred_pose = outputs["ego_n_to_ego_first"].float()
            oracle_pose = eval_batch["ego_n_to_ego_first"].float()
            record = {
                "sample_idx": sample_idx,
                "scene_id": batch["scene_id"],
                "clip_id": batch["clip_id"],
                "source_view_ids": batch["source_view_ids"],
                "target_view_ids": batch["target_view_ids"],
                "source_frame_ids_10hz": batch["source_frame_ids_10hz"],
                "online_sample_meta": batch.get("online_sample_meta"),
                "image_shape": list(images.shape),
                "load_summary": load_summary,
                "gt_scale_factor": gt_scale_factor,
                "pred_vs_gt": _pose_metrics(pred_pose, gt_metric),
                "oracle_gt_rel_vs_gt": _pose_metrics(oracle_pose, gt_metric),
                "pred_translation_norm_per_frame": [float(x) for x in pred_pose[0, :, :3, 3].norm(dim=-1).detach().cpu()],
                "gt_translation_norm_per_frame": [float(x) for x in gt_metric[0, :, :3, 3].norm(dim=-1).detach().cpu()],
                "relative_pose_translation_abs_mean_scaled_units": float(rel_pose_enc[..., :3].abs().mean().item()),
                "decoded_step_translation_norm_scaled_units": [float(x) for x in rel_pose_decoded[0, :, :3, 3].norm(dim=-1).detach().cpu()],
                "points_abs_mean_scaled_units": float(predictions["points"].abs().mean().item()),
                "points_conf_mean": float(predictions["points_conf"].mean().item()),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
