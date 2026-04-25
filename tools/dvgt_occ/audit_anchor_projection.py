#!/usr/bin/env python3
"""Audit DVGT point anchors by projecting them into source/target cameras."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.dvgt_occ.train import (
    build_dataset,
    build_online_dvgt_frontend,
    build_runtime_config,
    load_config,
    move_batch_to_device,
    prepare_batch_for_model,
    set_seed,
)
from dvgt_occ import DVGTOccConfig
from dvgt_occ.training import collate_dvgt_occ_batch
from dvgt.evaluation.utils.geometry import accumulate_transform_points_and_pose_to_first_frame
from dvgt.utils.pose_encoding import decode_pose

_RFU_FROM_RDF = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=torch.float32,
)
_RDF_FROM_RFU = _RFU_FROM_RDF.t().contiguous()


def _downsample_points(points: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    b, t, v, h, w, c = points.shape
    x = points.permute(0, 1, 2, 5, 3, 4).reshape(b * t * v, c, h, w)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x.reshape(b, t, v, c, size[0], size[1]).permute(0, 1, 2, 4, 5, 3).contiguous()


def _transform_point_axes(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(device=points.device, dtype=points.dtype)
    return points @ matrix.t()


def _project_coverage(
    points_first: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    height: int,
    width: int,
    radius: int,
) -> Dict[str, float]:
    pts = points_first.reshape(-1, 3).float()
    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=-1)
    view = torch.linalg.inv(camera_to_world.float())
    cam = (view @ pts_h.t()).t()[:, :3]
    z = cam[:, 2]
    valid_z = torch.isfinite(z) & (z > 0.5)
    fx, fy, cx, cy = [intrinsics[i].float() for i in range(4)]
    u = fx * cam[:, 0] / z.clamp_min(1e-6) + cx
    vv = fy * cam[:, 1] / z.clamp_min(1e-6) + cy
    valid = valid_z & torch.isfinite(u) & torch.isfinite(vv) & (u >= 0) & (u < width) & (vv >= 0) & (vv < height)
    mask = torch.zeros((height, width), device=pts.device, dtype=torch.bool)
    if torch.any(valid):
        ui = u[valid].round().long().clamp(0, width - 1)
        vi = vv[valid].round().long().clamp(0, height - 1)
        mask[vi, ui] = True
        if radius > 0:
            m = mask.float()[None, None]
            mask = F.max_pool2d(m, kernel_size=2 * radius + 1, stride=1, padding=radius)[0, 0] > 0
    z_valid = z[valid_z]
    u_valid = u[valid]
    v_valid = vv[valid]
    return {
        "point_count": int(pts.shape[0]),
        "z_valid_ratio": float(valid_z.float().mean().item()),
        "uv_valid_ratio": float(valid.float().mean().item()),
        "coverage_ratio": float(mask.float().mean().item()),
        "z_mean": float(z_valid.mean().item()) if z_valid.numel() else 0.0,
        "z_min": float(z_valid.min().item()) if z_valid.numel() else 0.0,
        "z_max": float(z_valid.max().item()) if z_valid.numel() else 0.0,
        "u_min": float(u_valid.min().item()) if u_valid.numel() else 0.0,
        "u_max": float(u_valid.max().item()) if u_valid.numel() else 0.0,
        "v_min": float(v_valid.min().item()) if v_valid.numel() else 0.0,
        "v_max": float(v_valid.max().item()) if v_valid.numel() else 0.0,
    }


def _make_runtime_args(config: Path, samples: int, seed: int, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        manifest=None,
        output_root=None,
        train_scene_ids=None,
        val_scene_ids=None,
        train_limit=samples,
        val_limit=None,
        max_steps=None,
        batch_size=1,
        num_workers=0,
        seed=seed,
        device=device,
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dvgt_occ/train_stage_c_gs_only_v59_online_dvgt_varvt_rfu2rdf_posepred.yaml"))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--downsample-height", type=int, default=56)
    parser.add_argument("--downsample-width", type=int, default=112)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-views", type=int, default=None)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--temporal-strides", type=int, nargs="*", default=None)
    parser.add_argument("--points-metric-scale", type=float, default=None)
    parser.add_argument("--anchor-mode", default=None, choices=("point_xyz", "ray_depth"))
    parser.add_argument("--out", type=Path, default=Path("data/nuscenes_dvgt_v0/logs/anchor_projection_audit_v59/anchor_projection.jsonl"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.image_height is not None or args.image_width is not None:
        model_overrides = dict(cfg.get("model", {}))
        if args.image_height is not None:
            model_overrides["image_height"] = int(args.image_height)
        if args.image_width is not None:
            model_overrides["image_width"] = int(args.image_width)
        cfg["model"] = model_overrides
    runtime = build_runtime_config(_make_runtime_args(args.config, args.samples, args.seed, args.device), cfg)
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
    if args.points_metric_scale is not None:
        runtime["online_points_metric_scale"] = float(args.points_metric_scale)
    if args.anchor_mode is not None:
        runtime["online_anchor_mode"] = str(args.anchor_mode)
    set_seed(int(runtime["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_cfg = replace(DVGTOccConfig(), **dict(cfg.get("model", {})))
    dataset = build_dataset(runtime["manifest"], runtime["output_root"], model_cfg, runtime["train_scene_ids"], args.samples, runtime)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_dvgt_occ_batch)
    frontend = build_online_dvgt_frontend(runtime, model_cfg, device)
    frontend.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for sample_idx, batch_cpu in enumerate(loader):
            batch = move_batch_to_device(batch_cpu, device)
            with torch.no_grad():
                frontend_out = frontend(batch["source_rgb"])
                points_raw = frontend_out["points"].float()
                rel_pose = frontend_out["relative_ego_pose_enc"].float()
                decoded, _ = decode_pose(rel_pose, pose_encoding_type="absT_quaR")
                decoded[..., :3, 3] *= float(runtime.get("online_points_metric_scale", 1.0))
                points_metric = points_raw * float(runtime.get("online_points_metric_scale", 1.0))
                pred_pose, points_pred_first = accumulate_transform_points_and_pose_to_first_frame(decoded, points_metric)

                runtime_pred = dict(runtime)
                runtime_pred["online_pose_mode"] = "pred"
                pred_batch = prepare_batch_for_model(dict(batch), frontend, runtime_pred)

                runtime_gt = dict(runtime)
                runtime_gt["online_pose_mode"] = "gt_debug"
                gt_batch = prepare_batch_for_model(dict(batch), frontend, runtime_gt)

                pred_points_1_4 = _downsample_points(pred_batch["points"], (args.downsample_height, args.downsample_width))
                gt_points_1_4 = _downsample_points(gt_batch["points"], (args.downsample_height, args.downsample_width))
                raw_points_1_4 = _downsample_points(points_metric, (args.downsample_height, args.downsample_width))

                target_view_ids = [int(x) for x in batch["target_view_ids"][0]]
                source_view_ids = [int(x) for x in batch["source_view_ids"][0]]
                source_index_by_view = {view_id: idx for idx, view_id in enumerate(source_view_ids)}
                target_is_source = batch["target_is_source"][0].detach().cpu().bool().tolist()

                records = []
                t = 0
                for target_idx, target_view_id in enumerate(target_view_ids):
                    all_pred = _project_coverage(
                        pred_points_1_4[0, t],
                        pred_batch["camera_to_world"][0, t, target_idx],
                        pred_batch["camera_intrinsics"][0, target_idx],
                        height=model_cfg.image_height,
                        width=model_cfg.image_width,
                        radius=args.radius,
                    )
                    all_gt = _project_coverage(
                        gt_points_1_4[0, t],
                        gt_batch["camera_to_world"][0, t, target_idx],
                        gt_batch["camera_intrinsics"][0, target_idx],
                        height=model_cfg.image_height,
                        width=model_cfg.image_width,
                        radius=args.radius,
                    )
                    item = {
                        "target_idx": target_idx,
                        "target_view_id": target_view_id,
                        "is_source": bool(target_is_source[target_idx]),
                        "all_source_anchors_pred_pose": all_pred,
                        "all_source_anchors_gt_pose": all_gt,
                    }
                    if target_view_id in source_index_by_view:
                        src_idx = source_index_by_view[target_view_id]
                        identity_camera = torch.eye(
                            4,
                            device=pred_batch["source_camera_to_world"].device,
                            dtype=pred_batch["source_camera_to_world"].dtype,
                        )
                        item["same_view_raw_ego_points_as_world"] = _project_coverage(
                            raw_points_1_4[0, t, src_idx:src_idx + 1],
                            pred_batch["source_camera_to_world"][0, t, src_idx],
                            pred_batch["source_camera_intrinsics"][0, src_idx],
                            height=model_cfg.image_height,
                            width=model_cfg.image_width,
                            radius=args.radius,
                        )
                        item["same_view_raw_points_identity_camera"] = _project_coverage(
                            raw_points_1_4[0, t, src_idx:src_idx + 1],
                            identity_camera,
                            pred_batch["source_camera_intrinsics"][0, src_idx],
                            height=model_cfg.image_height,
                            width=model_cfg.image_width,
                            radius=args.radius,
                        )
                        item["same_view_pred_pose"] = _project_coverage(
                            pred_points_1_4[0, t, src_idx:src_idx + 1],
                            pred_batch["source_camera_to_world"][0, t, src_idx],
                            pred_batch["source_camera_intrinsics"][0, src_idx],
                            height=model_cfg.image_height,
                            width=model_cfg.image_width,
                            radius=args.radius,
                        )
                        item["same_view_gt_pose"] = _project_coverage(
                            gt_points_1_4[0, t, src_idx:src_idx + 1],
                            gt_batch["gt_source_camera_to_first_ego"][0, t, src_idx],
                            gt_batch["source_camera_intrinsics"][0, src_idx],
                            height=model_cfg.image_height,
                            width=model_cfg.image_width,
                            radius=args.radius,
                        )
                        item["same_view_gt_pose_points_rfu_to_rdf"] = _project_coverage(
                            _transform_point_axes(gt_points_1_4[0, t, src_idx:src_idx + 1], _RDF_FROM_RFU),
                            gt_batch["gt_source_camera_to_first_ego"][0, t, src_idx],
                            gt_batch["source_camera_intrinsics"][0, src_idx],
                            height=model_cfg.image_height,
                            width=model_cfg.image_width,
                            radius=args.radius,
                        )
                        item["same_view_gt_pose_points_rdf_to_rfu"] = _project_coverage(
                            _transform_point_axes(gt_points_1_4[0, t, src_idx:src_idx + 1], _RFU_FROM_RDF),
                            gt_batch["gt_source_camera_to_first_ego"][0, t, src_idx],
                            gt_batch["source_camera_intrinsics"][0, src_idx],
                            height=model_cfg.image_height,
                            width=model_cfg.image_width,
                            radius=args.radius,
                        )
                    records.append(item)

                pose_err = (pred_pose[..., :3, 3] - batch["gt_ego_n_to_first_ego"].float()[..., :3, 3]).norm(dim=-1)
                out_record = {
                    "sample_idx": sample_idx,
                    "scene_id": batch["scene_id"],
                    "clip_id": batch["clip_id"],
                    "source_view_ids": source_view_ids,
                    "target_view_ids": target_view_ids,
                    "frame_ids_10hz": batch["frame_ids_10hz"],
                    "online_sample_meta": batch.get("online_sample_meta"),
                    "pose_trans_err_mean": float(pose_err.mean().item()),
                    "points_raw_abs_mean": float(points_raw.abs().mean().item()),
                    "points_metric_abs_mean": float(points_metric.abs().mean().item()),
                    "records_frame0": records,
                }
                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
