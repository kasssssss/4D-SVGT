#!/usr/bin/env python3
"""Stage-C/D training entry for DVGT-Occ."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import socket
import time
from datetime import timedelta
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ import DEFAULT_DVGT_OCC_CONFIG, DVGTOccConfig
from dvgt_occ.data import DVGTOccClipDataset
from dvgt_occ.data.manifest import load_manifest
from dvgt_occ.models import DVGTOccModel
from dvgt_occ.training import (
    DEFAULT_CACHE_KEYS,
    DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS,
    DEFAULT_SUPERVISION_KEYS,
    DVGTOccLossBuilder,
    LossWeights,
    binary_iou_from_logits,
    binary_stats_from_logits,
    collate_dvgt_occ_batch,
    iou_threshold_sweep_from_logits,
    masked_l1,
    masked_psnr,
    move_batch_to_device,
    reduce_metrics,
    render_valid_mask_from_points_conf,
    resolve_loss_weights,
    soft_iou_from_logits,
    save_training_visualization,
    stage_b_after_stability_weights,
    stage_c_bridge_warmup_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-C/D training for DVGT-Occ.")
    parser.add_argument("--config", type=Path, default=Path("configs/dvgt_occ/train_stage_c_small.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--train-scene-ids", nargs="*", default=None)
    parser.add_argument("--val-scene-ids", nargs="*", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--ddp-timeout-sec", type=int, default=None)
    parser.add_argument("--ddp-backend", type=str, default=None)
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--broadcast-buffers", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed(timeout_sec: int, backend_override: str | None = None) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = backend_override or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_sec))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, rank, world_size, local_rank


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(args: argparse.Namespace, ddp: bool, local_rank: int) -> torch.device:
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank if ddp else 0}")
    return torch.device("cpu")


def build_runtime_config(args: argparse.Namespace, cfg: Dict[str, object]) -> Dict[str, object]:
    data_cfg = dict(cfg.get("data", {}))
    train_cfg = dict(cfg.get("train", {}))
    ddp_cfg = dict(cfg.get("ddp", {}))
    ready_files = [Path(path) for path in data_cfg.get("supervision_ready_files", [])]
    runtime = {
        "manifest": Path(args.manifest or data_cfg.get("manifest", "data/nuscenes_dvgt_v0/manifest_trainval.json")),
        "output_root": Path(args.output_root or data_cfg.get("output_root", "data/nuscenes_dvgt_v0")),
        "train_scene_ids": args.train_scene_ids if args.train_scene_ids is not None else data_cfg.get("train_scene_ids"),
        "val_scene_ids": args.val_scene_ids if args.val_scene_ids is not None else data_cfg.get("val_scene_ids"),
        "train_limit": args.train_limit if args.train_limit is not None else data_cfg.get("train_limit"),
        "val_limit": args.val_limit if args.val_limit is not None else data_cfg.get("val_limit"),
        "batch_size": int(args.batch_size or train_cfg.get("batch_size", 1)),
        "num_workers": int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)),
        "max_steps": int(args.max_steps or train_cfg.get("max_steps", 20000)),
        "seed": int(args.seed or train_cfg.get("seed", 42)),
        "log_interval": int(args.log_interval if args.log_interval is not None else train_cfg.get("log_interval", 100)),
        "save_interval": int(args.save_interval if args.save_interval is not None else train_cfg.get("save_interval", 2000)),
        "val_interval": int(args.val_interval if args.val_interval is not None else train_cfg.get("val_interval", 2000)),
        "val_batches": int(args.val_batches if args.val_batches is not None else train_cfg.get("val_batches", 32)),
        "val_dynamic_only": bool(data_cfg.get("val_dynamic_only", False)),
        "val_min_dynamic_pixels": int(data_cfg.get("val_min_dynamic_pixels", 32)),
        "dynamic_val_scene_ids": data_cfg.get("dynamic_val_scene_ids"),
        "dynamic_val_limit": data_cfg.get("dynamic_val_limit"),
        "dynamic_val_only": bool(data_cfg.get("dynamic_val_only", bool(data_cfg.get("dynamic_val_scene_ids")))),
        "dynamic_val_min_dynamic_pixels": int(data_cfg.get("dynamic_val_min_dynamic_pixels", data_cfg.get("val_min_dynamic_pixels", 32))),
        "dynamic_val_batches": int(train_cfg.get("dynamic_val_batches", args.val_batches if args.val_batches is not None else train_cfg.get("val_batches", 32))),
        "grad_clip": float(train_cfg.get("grad_clip", 1.0)),
        "amp": bool(train_cfg.get("amp", True)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
        "warmup_iters": int(train_cfg.get("warmup_iters", 1500)),
        "min_lr": float(train_cfg.get("min_lr", 1e-6)),
        "require_supervision_frozen": bool(data_cfg.get("require_supervision_frozen", False)),
        "supervision_ready_files": ready_files,
        "bridge_start_step": int(train_cfg.get("bridge_start_step", train_cfg.get("stability_start_step", 0))),
        "stability_start_step": int(train_cfg.get("stability_start_step", 0)),
        "allow_tf32": bool(train_cfg.get("allow_tf32", True)),
        "log_dir": Path(args.log_dir or train_cfg.get("log_dir", "data/nuscenes_dvgt_v0/logs/train_stage")),
        "visualize_every_epochs": int(train_cfg.get("visualize_every_epochs", 100)),
        "visualize_frame_index": int(train_cfg.get("visualize_frame_index", 0)),
        "visualize_view_index": int(train_cfg.get("visualize_view_index", 0)),
        "visualize_sample_index": int(train_cfg.get("visualize_sample_index", 0)),
        "visualize_max_scene_points": int(train_cfg.get("visualize_max_scene_points", 20000)),
        "find_unused_parameters": bool(args.find_unused_parameters or ddp_cfg.get("find_unused_parameters", False)),
        "broadcast_buffers": bool(args.broadcast_buffers or ddp_cfg.get("broadcast_buffers", False)),
        "static_graph": bool(ddp_cfg.get("static_graph", False)),
        "ddp_timeout_sec": int(args.ddp_timeout_sec or ddp_cfg.get("timeout_sec", 1800)),
        "ddp_backend": str(args.ddp_backend or ddp_cfg.get("backend", "nccl")),
        "resume": None if args.no_resume else (Path(args.resume or train_cfg.get("resume", "")) if (args.resume or train_cfg.get("resume")) else None),
    }
    return runtime


def build_model_config(cfg: Dict[str, object]) -> DVGTOccConfig:
    model_cfg = dict(cfg.get("model", {}))
    base = DEFAULT_DVGT_OCC_CONFIG
    values = {
        "batch_size": int(model_cfg.get("batch_size", base.batch_size)),
        "num_frames": int(model_cfg.get("num_frames", base.num_frames)),
        "num_views": int(model_cfg.get("num_views", base.num_views)),
        "image_height": int(model_cfg.get("image_height", base.image_height)),
        "image_width": int(model_cfg.get("image_width", base.image_width)),
        "patch_size": int(model_cfg.get("patch_size", base.patch_size)),
        "token_dim": int(model_cfg.get("token_dim", base.token_dim)),
        "agg_token_dim": int(model_cfg.get("agg_token_dim", base.agg_token_dim)),
        "neck_dim": int(model_cfg.get("neck_dim", base.neck_dim)),
        "full_dim": int(model_cfg.get("full_dim", base.full_dim)),
        "dynamic_query_dim": int(model_cfg.get("dynamic_query_dim", base.dynamic_query_dim)),
        "instance_dim": int(model_cfg.get("instance_dim", base.instance_dim)),
        "motion_dim": int(model_cfg.get("motion_dim", base.motion_dim)),
        "dynamic_classes": int(model_cfg.get("dynamic_classes", base.dynamic_classes)),
        "semantic_classes": int(model_cfg.get("semantic_classes", base.semantic_classes)),
        "projected_semantic_classes": int(model_cfg.get("projected_semantic_classes", base.projected_semantic_classes)),
        "max_track_queries": int(model_cfg.get("max_track_queries", base.max_track_queries)),
        "new_queries": int(model_cfg.get("new_queries", base.new_queries)),
        "sparse_dynamic_anchors": int(model_cfg.get("sparse_dynamic_anchors", base.sparse_dynamic_anchors)),
        "global_latents": int(model_cfg.get("global_latents", base.global_latents)),
        "occ_samples_per_scale": int(model_cfg.get("occ_samples_per_scale", base.occ_samples_per_scale)),
        "gs_bias_scale": float(model_cfg.get("gs_bias_scale", base.gs_bias_scale)),
        "render_splat_radius": int(model_cfg.get("render_splat_radius", base.render_splat_radius)),
        "selected_layers": tuple(model_cfg.get("selected_layers", list(base.selected_layers))),
        "camera_view_ids": tuple(model_cfg.get("camera_view_ids", list(base.camera_view_ids))),
        "occ_grid": base.occ_grid,
    }
    return DVGTOccConfig(**values)


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, object]) -> torch.optim.Optimizer:
    opt_cfg = dict(cfg.get("optimizer", {}))
    base_lr = float(opt_cfg.get("base_lr", 2e-4))
    gs_lr = float(opt_cfg.get("gs_lr", 1e-4))
    bridge_lr = float(opt_cfg.get("bridge_lr", 1e-4))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))

    groups = {
        "base": {"params": [], "lr": base_lr},
        "gs": {"params": [], "lr": gs_lr},
        "bridge": {"params": [], "lr": bridge_lr},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("gs_head."):
            groups["gs"]["params"].append(param)
        elif "global_latent_bridge" in name or "gs_occ_global_latent_bridge" in name:
            groups["bridge"]["params"].append(param)
        else:
            groups["base"]["params"].append(param)
    return torch.optim.AdamW([group for group in groups.values() if group["params"]], betas=betas, weight_decay=weight_decay)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, step: int, max_steps: int, warmup_iters: int, min_lr: float) -> None:
    for group in optimizer.param_groups:
        base_lr = group.setdefault("initial_lr", group["lr"])
        if step < warmup_iters:
            lr = base_lr * float(step + 1) / float(max(warmup_iters, 1))
        else:
            progress = float(step - warmup_iters) / float(max(max_steps - warmup_iters, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            lr = min_lr + (base_lr - min_lr) * cosine
        group["lr"] = lr


def build_loss_weights(cfg: Dict[str, object]) -> tuple[LossWeights, LossWeights, LossWeights]:
    weights_cfg = dict(cfg.get("loss_weights", {}))
    merged = DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS.to_dict()
    merged.update(weights_cfg.get("base", weights_cfg))
    base = LossWeights(**merged)

    bridge_cfg = weights_cfg.get("bridge_warmup")
    bridge = stage_c_bridge_warmup_weights()
    if bridge_cfg:
        bridge_merged = bridge.to_dict()
        bridge_merged.update(bridge_cfg)
        bridge = LossWeights(**bridge_merged)

    after_cfg = weights_cfg.get("after_stability")
    after = stage_b_after_stability_weights()
    if after_cfg:
        after_merged = after.to_dict()
        after_merged.update(after_cfg)
        after = LossWeights(**after_merged)
    return base, bridge, after


def discover_scene_splits(manifest_path: Path, train_scene_ids: Sequence[str] | None, val_scene_ids: Sequence[str] | None) -> tuple[List[str] | None, List[str] | None]:
    if train_scene_ids:
        return list(train_scene_ids), list(val_scene_ids or [])
    if val_scene_ids:
        entries = load_manifest(manifest_path)
        all_scenes = sorted({str(entry["scene_id"]) for entry in entries})
        val_set = {str(scene_id) for scene_id in val_scene_ids}
        train = [scene_id for scene_id in all_scenes if scene_id not in val_set]
        return train, list(val_scene_ids)
    return None, None


def build_dataset(
    manifest: Path,
    root: Path,
    model_cfg: DVGTOccConfig,
    scene_ids: Sequence[str] | None,
    limit: int | None,
) -> DVGTOccClipDataset:
    return DVGTOccClipDataset(
        manifest_path=manifest,
        root=root,
        load_cache=True,
        load_supervision=True,
        load_scene_sam3_full=True,
        full_res_size=(model_cfg.image_height, model_cfg.image_width),
        projected_semantic_classes=model_cfg.projected_semantic_classes,
        cache_keys=DEFAULT_CACHE_KEYS,
        supervision_keys=DEFAULT_SUPERVISION_KEYS,
        scene_ids=scene_ids,
        limit=limit,
    )


def select_dynamic_subset(dataset: DVGTOccClipDataset, min_dynamic_pixels: int) -> DVGTOccClipDataset | Subset:
    keep_indices: list[int] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        sam3 = sample.get("sam3_fullres")
        if sam3 is None:
            continue
        dyn = sam3["sam3_dyn_mask_full"]
        valid = sam3.get("sam3_valid_mask_full")
        if valid is not None:
            dyn_pixels = int(((dyn > 0.5) & (valid > 0.5)).sum())
        else:
            dyn_pixels = int((dyn > 0.5).sum())
        if dyn_pixels >= int(min_dynamic_pixels):
            keep_indices.append(idx)
    if keep_indices:
        return Subset(dataset, keep_indices)
    return dataset


def build_loader(dataset: DVGTOccClipDataset, batch_size: int, num_workers: int, ddp_enabled: bool, rank: int, world_size: int, shuffle: bool) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if ddp_enabled:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_dvgt_occ_batch,
    )
    return loader, sampler


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _tensor_nonfinite_summary(name: str, tensor: torch.Tensor) -> Dict[str, object] | None:
    if tensor.numel() == 0:
        return None
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return None
    total = int(tensor.numel())
    finite_count = int(finite.sum().item())
    payload: Dict[str, object] = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite_count": finite_count,
        "nonfinite_count": total - finite_count,
    }
    finite_vals = tensor[finite]
    if finite_vals.numel() > 0:
        payload["finite_min"] = float(finite_vals.min().item())
        payload["finite_max"] = float(finite_vals.max().item())
    return payload


def _collect_nonfinite_tensors(obj: object, prefix: str = "") -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    if isinstance(obj, torch.Tensor):
        summary = _tensor_nonfinite_summary(prefix or "tensor", obj.detach())
        if summary is not None:
            summaries.append(summary)
        return summaries
    if is_dataclass(obj):
        for key, value in obj.__dict__.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            summaries.extend(_collect_nonfinite_tensors(value, child_prefix))
        return summaries
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            summaries.extend(_collect_nonfinite_tensors(value, child_prefix))
        return summaries
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            summaries.extend(_collect_nonfinite_tensors(value, child_prefix))
        return summaries
    return summaries


def _collect_nonfinite_gradients(model: torch.nn.Module) -> List[Dict[str, object]]:
    bad: List[Dict[str, object]] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        summary = _tensor_nonfinite_summary(name, param.grad.detach())
        if summary is not None:
            bad.append(summary)
    return bad


def write_status_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def to_jsonable(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return obj


def validate_supervision_contract(runtime: Dict[str, object]) -> None:
    if not runtime["require_supervision_frozen"]:
        return
    missing = [str(path) for path in runtime["supervision_ready_files"] if not Path(path).exists()]
    if missing:
        joined = "\n".join(missing)
        raise RuntimeError(f"Supervision freeze contract not satisfied. Missing required files:\n{joined}")


def save_checkpoint(path: Path, step: int, raw_model: torch.nn.Module, optimizer: torch.optim.Optimizer, score: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"step": step, "model": raw_model.state_dict(), "optimizer": optimizer.state_dict()}
    if score is not None:
        payload["score"] = score
    torch.save(payload, path)


def collect_debug_metrics(outputs: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return collect_debug_metrics_with_batch(outputs, None)


def collect_debug_metrics_with_batch(outputs: Dict[str, object], batch: Dict[str, torch.Tensor] | None) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}
    assignment = outputs.get("assignment")
    if assignment is not None:
        metrics["debug_assign_bg_mean"] = assignment.background_prob.mean()
        metrics["debug_assign_bg_gt_05_ratio"] = (assignment.background_prob > 0.5).float().mean()
        metrics["debug_assign_fg_max_mean"] = assignment.assignment_prob[..., 1:].max(dim=-1).values.mean()
        metrics["debug_assign_pos_ratio"] = (assignment.assigned_query >= 0).float().mean()
        metrics["debug_assign_gate_active"] = (assignment.local_gate > 0.5).float().mean()
        metrics["debug_assign_unique_query_mean"] = torch.stack(
            [
                assignment.assigned_query[batch_idx][assignment.assigned_query[batch_idx] >= 0].unique().numel()
                if bool((assignment.assigned_query[batch_idx] >= 0).any().item())
                else assignment.assigned_query.new_tensor(0)
                for batch_idx in range(assignment.assigned_query.shape[0])
            ]
        ).float().mean()
        if assignment.feature_similarity is not None:
            metrics["debug_assign_feat_sim_mean"] = assignment.feature_similarity.mean()
        if assignment.routing_keep_score is not None:
            metrics["debug_assign_routing_keep_mean"] = assignment.routing_keep_score.mean()

    gaussians = outputs.get("gaussians")
    if gaussians is not None:
        metrics["debug_gauss_keep_mean"] = gaussians.keep_score.mean()
        metrics["debug_gauss_keep_gt_01_ratio"] = (gaussians.keep_score > 0.1).float().mean()
        metrics["debug_gauss_keep_gt_05_ratio"] = (gaussians.keep_score > 0.5).float().mean()
        metrics["debug_gauss_effective_count_bt_mean"] = (gaussians.keep_score > 0.1).float().reshape(gaussians.keep_score.shape[0], gaussians.keep_score.shape[1], -1).sum(dim=-1).mean()
        metrics["debug_gauss_opacity_mean"] = gaussians.opacity.mean()
        metrics["debug_gauss_scale_mean"] = gaussians.scale.mean()
        if batch is not None and "gs_global_track_id_1_8" in batch:
            supervised = (batch["gs_global_track_id_1_8"] >= 0).float()
            keep = (gaussians.keep_score.squeeze(-1) > 0.1).float()
            supervised_denom = supervised.sum().clamp_min(1.0)
            metrics["debug_gauss_supervised_ratio"] = supervised.mean()
            metrics["debug_gauss_supervised_kept_ratio"] = (keep * supervised).sum() / supervised_denom
            metrics["debug_global_weak_valid_ratio"] = supervised.mean()

    render = outputs.get("render")
    if render is not None:
        metrics["debug_render_alpha_coverage"] = (render.render_alpha_all > 1e-3).float().mean()
        metrics["debug_render_alpha_mean"] = render.render_alpha_all.mean()
        if render.debug_mean_sigma is not None:
            metrics["debug_render_sigma_mean"] = render.debug_mean_sigma
        if render.debug_touch_ratio is not None:
            metrics["debug_render_touch_ratio"] = render.debug_touch_ratio
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loss_builder: DVGTOccLossBuilder,
    loader: DataLoader,
    device: torch.device,
    ddp_enabled: bool,
    max_batches: int,
    prefix: str = "val",
) -> Dict[str, float]:
    model.eval()
    loss_totals: Dict[str, torch.Tensor] = {}
    denom = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        outputs["step_hint"] = 0
        loss_total, loss_dict = loss_builder(outputs, batch)
        occ_logits = outputs["occ"].occ_logit
        occ_target = batch["occ_label"]
        dyn_logits = torch.logit(outputs["render"].render_alpha_dynamic.clamp(1e-4, 1.0 - 1e-4))
        dyn_target = batch["sam3_dyn_mask_full"].unsqueeze(3)
        dyn_valid = batch.get("sam3_valid_mask_full")
        if dyn_valid is not None:
            dyn_valid = dyn_valid.unsqueeze(3)
        render_valid = render_valid_mask_from_points_conf(batch["points_conf"])
        metrics = {
            f"{prefix}_loss_total": loss_total.detach(),
            f"{prefix}_occ_iou": binary_iou_from_logits(occ_logits, occ_target),
            f"{prefix}_occ_soft_iou": soft_iou_from_logits(occ_logits, occ_target),
            f"{prefix}_dyn_mask_iou": binary_iou_from_logits(dyn_logits, dyn_target, valid_mask=dyn_valid),
            f"{prefix}_render_l1": masked_l1(outputs["render"].render_rgb_all, batch["rgb_target"], render_valid),
            f"{prefix}_render_psnr": masked_psnr(outputs["render"].render_rgb_all, batch["rgb_target"], render_valid),
        }
        for key, value in binary_stats_from_logits(occ_logits, occ_target).items():
            metrics[f"{prefix}_occ_{key}"] = value
        for key, value in iou_threshold_sweep_from_logits(occ_logits, occ_target).items():
            metrics[f"{prefix}_occ_{key}"] = value
        for key, value in binary_stats_from_logits(dyn_logits, dyn_target, valid_mask=dyn_valid).items():
            metrics[f"{prefix}_dyn_mask_{key}"] = value
        for key, value in iou_threshold_sweep_from_logits(dyn_logits, dyn_target, valid_mask=dyn_valid).items():
            metrics[f"{prefix}_dyn_mask_{key}"] = value
        for key, value in loss_dict.items():
            metrics[f"{prefix}_loss_{key}"] = value.detach()
        for key, value in collect_debug_metrics_with_batch(outputs, batch).items():
            metrics[f"{prefix}_{key}"] = value.detach()
        for key, value in metrics.items():
            loss_totals[key] = loss_totals.get(key, torch.zeros_like(value)) + value
        denom += 1
    if denom == 0:
        raise RuntimeError("Validation loader yielded no batches.")
    averaged = {key: value / float(denom) for key, value in loss_totals.items()}
    model.train()
    return reduce_metrics(averaged, ddp_enabled)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = build_runtime_config(args, cfg)
    ddp_enabled, rank, world_size, local_rank = init_distributed(
        timeout_sec=runtime["ddp_timeout_sec"],
        backend_override=runtime["ddp_backend"],
    )
    run_start_time = time.time()
    run_exception: BaseException | None = None
    try:
        model_cfg = build_model_config(cfg)
        set_seed(runtime["seed"] + rank)
        device = resolve_device(args, ddp_enabled, local_rank)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = runtime["allow_tf32"]
            torch.backends.cudnn.allow_tf32 = runtime["allow_tf32"]

        validate_supervision_contract(runtime)

        train_scene_ids, val_scene_ids = discover_scene_splits(runtime["manifest"], runtime["train_scene_ids"], runtime["val_scene_ids"])
        train_dataset = build_dataset(runtime["manifest"], runtime["output_root"], model_cfg, train_scene_ids, runtime["train_limit"])
        val_dataset = build_dataset(runtime["manifest"], runtime["output_root"], model_cfg, val_scene_ids, runtime["val_limit"]) if val_scene_ids else None
        if val_dataset is not None and runtime["val_dynamic_only"]:
            val_dataset = select_dynamic_subset(val_dataset, runtime["val_min_dynamic_pixels"])
        dynamic_val_scene_ids = runtime["dynamic_val_scene_ids"]
        dynamic_val_dataset = None
        if runtime["dynamic_val_only"] and (dynamic_val_scene_ids is not None or val_scene_ids is not None):
            dynamic_val_dataset = build_dataset(
                runtime["manifest"],
                runtime["output_root"],
                model_cfg,
                dynamic_val_scene_ids if dynamic_val_scene_ids is not None else val_scene_ids,
                runtime["dynamic_val_limit"],
            )
            dynamic_val_dataset = select_dynamic_subset(dynamic_val_dataset, runtime["dynamic_val_min_dynamic_pixels"])
        if len(train_dataset) == 0:
            raise RuntimeError("No training clips selected.")

        train_loader, train_sampler = build_loader(
            train_dataset, runtime["batch_size"], runtime["num_workers"], ddp_enabled, rank, world_size, shuffle=True
        )
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader, _ = build_loader(
                val_dataset, runtime["batch_size"], runtime["num_workers"], ddp_enabled, rank, world_size, shuffle=False
            )
        dynamic_val_loader = None
        if dynamic_val_dataset is not None and len(dynamic_val_dataset) > 0:
            dynamic_val_loader, _ = build_loader(
                dynamic_val_dataset, runtime["batch_size"], runtime["num_workers"], ddp_enabled, rank, world_size, shuffle=False
            )

        model = DVGTOccModel(config=model_cfg).to(device)
        raw_model = model
        if runtime["gradient_checkpointing"] and hasattr(raw_model, "enable_gradient_checkpointing"):
            raw_model.enable_gradient_checkpointing(True)
        if ddp_enabled:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=runtime["find_unused_parameters"],
                broadcast_buffers=runtime["broadcast_buffers"],
                static_graph=runtime["static_graph"],
            )
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model

        base_weights, bridge_weights, after_weights = build_loss_weights(cfg)
        loss_builder = DVGTOccLossBuilder(config=model_cfg, weights=base_weights).to(device)
        optimizer = build_optimizer(raw_model, cfg)
        autocast_enabled = bool(runtime["amp"] and device.type == "cuda")

        start_step = 0
        if runtime["resume"] is not None:
            checkpoint = torch.load(runtime["resume"], map_location=device)
            raw_model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint.get("step", -1)) + 1

        log_dir = runtime["log_dir"]
        if rank == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {
                "runtime": to_jsonable(runtime),
                "model": asdict(model_cfg),
                "train_scene_ids": train_scene_ids,
                "val_scene_ids": val_scene_ids,
                "val_dynamic_only": runtime["val_dynamic_only"],
                "val_min_dynamic_pixels": runtime["val_min_dynamic_pixels"],
                "dynamic_val_scene_ids": dynamic_val_scene_ids if dynamic_val_scene_ids is not None else val_scene_ids,
                "dynamic_val_only": runtime["dynamic_val_only"],
                "dynamic_val_min_dynamic_pixels": runtime["dynamic_val_min_dynamic_pixels"],
                "base_loss_weights": base_weights.to_dict(),
                "bridge_warmup_loss_weights": bridge_weights.to_dict(),
                "after_stability_loss_weights": after_weights.to_dict(),
                "ddp": {
                    "enabled": ddp_enabled,
                    "world_size": world_size,
                    "backend": runtime["ddp_backend"],
                    "timeout_sec": runtime["ddp_timeout_sec"],
                    "find_unused_parameters": runtime["find_unused_parameters"],
                    "broadcast_buffers": runtime["broadcast_buffers"],
                },
            }
            (log_dir / "train_config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            write_status_json(
                log_dir / "run_status.json",
                {
                    "state": "running",
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "start_time": run_start_time,
                    "last_step": start_step - 1,
                    "last_val_step": None,
                },
            )

            def _handle_signal(signum: int, _frame: object) -> None:
                write_status_json(
                    log_dir / "run_status.json",
                    {
                        "state": "signal_exit",
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "start_time": run_start_time,
                        "last_step": last_train_record["step"] if "last_train_record" in locals() and last_train_record is not None else start_step - 1,
                        "last_val_step": last_val_record["step"] if "last_val_record" in locals() and last_val_record is not None else None,
                        "signal": int(signum),
                        "heartbeat_time": time.time(),
                    },
                )
                raise SystemExit(128 + int(signum))

            signal.signal(signal.SIGTERM, _handle_signal)
            signal.signal(signal.SIGINT, _handle_signal)

        model.train()
        train_iter = iter(train_loader)
        best_score = None
        last_train_record: dict[str, object] | None = None
        last_val_record: dict[str, object] | None = None
        skipped_nonfinite_steps = 0
        epoch_size = max(len(train_loader), 1)
        for step in range(start_step, runtime["max_steps"]):
            if train_sampler is not None and step % max(len(train_loader), 1) == 0:
                train_sampler.set_epoch(step)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = move_batch_to_device(batch, device)

            loss_builder.weights = resolve_loss_weights(
                step=step,
                base=base_weights,
                bridge_warmup=bridge_weights,
                after_stability=after_weights,
                bridge_start_step=runtime["bridge_start_step"],
                stability_start_step=runtime["stability_start_step"],
            )
            adjust_learning_rate(optimizer, step, runtime["max_steps"], runtime["warmup_iters"], runtime["min_lr"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, enabled=autocast_enabled):
                outputs = model(batch)
                outputs["step_hint"] = step
                loss_total, loss_dict = loss_builder(outputs, batch)
            if not bool(torch.isfinite(loss_total).item()):
                if rank == 0:
                    payload = {
                        "step": step,
                        "reason": "nonfinite_loss",
                        "scene_id": batch["scene_id"],
                        "clip_id": batch["clip_id"],
                        "loss_total": str(loss_total.detach().item()),
                        "bad_losses": {
                            key: str(value.detach().item())
                            for key, value in loss_dict.items()
                            if not bool(torch.isfinite(value.detach()).item())
                        },
                        "batch_nonfinite": _collect_nonfinite_tensors(batch, prefix="batch"),
                        "output_nonfinite": _collect_nonfinite_tensors(outputs, prefix="outputs"),
                    }
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
                    append_jsonl(log_dir / "nonfinite_steps.jsonl", payload)
                skipped_nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            loss_total.backward()
            bad_grads = _collect_nonfinite_gradients(raw_model)
            if bad_grads:
                if rank == 0:
                    payload = {
                        "step": step,
                        "reason": "nonfinite_grad",
                        "scene_id": batch["scene_id"],
                        "clip_id": batch["clip_id"],
                        "loss_total": float(loss_total.detach().item()),
                        "bad_grads": bad_grads,
                    }
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
                    append_jsonl(log_dir / "nonfinite_steps.jsonl", payload)
                skipped_nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), runtime["grad_clip"])
            if not math.isfinite(float(grad_norm)):
                if rank == 0:
                    payload = {
                        "step": step,
                        "reason": "nonfinite_grad_norm",
                        "scene_id": batch["scene_id"],
                        "clip_id": batch["clip_id"],
                        "loss_total": float(loss_total.detach().item()),
                        "grad_norm": str(float(grad_norm)),
                    }
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
                    append_jsonl(log_dir / "nonfinite_steps.jsonl", payload)
                skipped_nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

            if rank == 0:
                last_train_record = {
                    "step": step,
                    "loss_total": float(loss_total.detach().item()),
                    "lr_base": optimizer.param_groups[0]["lr"],
                    "scene_id": batch["scene_id"],
                    "clip_id": batch["clip_id"],
                    "loss_weights": loss_builder.weights.to_dict(),
                    "curriculum_phase": (
                        "base"
                        if step < runtime["bridge_start_step"]
                        else ("bridge_warmup" if step < runtime["stability_start_step"] else "joint_refine")
                    ),
                }
                last_train_record.update({f"loss_{key}": float(value.detach().item()) for key, value in loss_dict.items()})
                last_train_record.update(
                    {key: float(value.detach().item()) for key, value in collect_debug_metrics_with_batch(outputs, batch).items()}
                )
                last_train_record["skipped_nonfinite_steps"] = skipped_nonfinite_steps
                if step % runtime["log_interval"] == 0 or step == runtime["max_steps"] - 1:
                    print(json.dumps(last_train_record, ensure_ascii=False), flush=True)
                    append_jsonl(log_dir / "train_metrics.jsonl", last_train_record)
                    write_status_json(
                        log_dir / "run_status.json",
                        {
                            "state": "running",
                            "hostname": socket.gethostname(),
                            "pid": os.getpid(),
                            "start_time": run_start_time,
                            "last_step": step,
                            "last_val_step": last_val_record["step"] if last_val_record is not None else None,
                            "heartbeat_time": time.time(),
                        },
                    )

            if rank == 0 and runtime["visualize_every_epochs"] > 0:
                visualize_stride = runtime["visualize_every_epochs"] * epoch_size
                should_visualize = ((step + 1) % visualize_stride == 0) or (step == runtime["max_steps"] - 1)
                if should_visualize:
                    epoch = (step + 1) // epoch_size
                    vis_path = log_dir / "visuals" / f"epoch_{epoch:04d}_step_{step + 1:06d}.png"
                    save_training_visualization(
                        batch,
                        outputs,
                        model_cfg,
                        vis_path,
                        step=step,
                        epoch=epoch,
                        sample_idx=runtime["visualize_sample_index"],
                        frame_idx=runtime["visualize_frame_index"],
                        view_idx=runtime["visualize_view_index"],
                        max_scene_points=runtime["visualize_max_scene_points"],
                    )

            if rank == 0 and runtime["save_interval"] > 0 and (step + 1) % runtime["save_interval"] == 0:
                save_checkpoint(log_dir / f"step_{step + 1:06d}.pt", step, raw_model, optimizer)
                save_checkpoint(log_dir / "last.pt", step, raw_model, optimizer, score=best_score)

            if (val_loader is not None or dynamic_val_loader is not None) and runtime["val_interval"] > 0 and (step + 1) % runtime["val_interval"] == 0:
                val_metrics: Dict[str, float] = {}
                if val_loader is not None:
                    val_metrics.update(evaluate(model, loss_builder, val_loader, device, ddp_enabled, runtime["val_batches"], prefix="val"))
                if dynamic_val_loader is not None:
                    val_metrics.update(
                        evaluate(
                            model,
                            loss_builder,
                            dynamic_val_loader,
                            device,
                            ddp_enabled,
                            runtime["dynamic_val_batches"],
                            prefix="dynheavy_val",
                        )
                    )
                if rank == 0:
                    val_metrics["step"] = step
                    last_val_record = dict(val_metrics)
                    print(json.dumps(val_metrics, ensure_ascii=False), flush=True)
                    append_jsonl(log_dir / "val_metrics.jsonl", val_metrics)
                    score = val_metrics.get("val_occ_iou", 0.0) + val_metrics.get("val_dyn_mask_iou", 0.0)
                    if best_score is None or score > best_score:
                        best_score = score
                        save_checkpoint(log_dir / "best.pt", step, raw_model, optimizer, score=score)
                    save_checkpoint(log_dir / "last.pt", step, raw_model, optimizer, score=best_score)
                    write_status_json(
                        log_dir / "run_status.json",
                        {
                            "state": "running",
                            "hostname": socket.gethostname(),
                            "pid": os.getpid(),
                            "start_time": run_start_time,
                            "last_step": step,
                            "last_val_step": step,
                            "heartbeat_time": time.time(),
                            "best_score": best_score,
                        },
                    )

        if rank == 0:
            if last_train_record is not None and not (log_dir / "train_metrics.jsonl").exists():
                append_jsonl(log_dir / "train_metrics.jsonl", last_train_record)
            if last_val_record is not None and not (log_dir / "val_metrics.jsonl").exists():
                append_jsonl(log_dir / "val_metrics.jsonl", last_val_record)
            save_checkpoint(log_dir / "last.pt", runtime["max_steps"] - 1, raw_model, optimizer, score=best_score)
            write_status_json(
                log_dir / "run_status.json",
                {
                    "state": "finished",
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "start_time": run_start_time,
                    "finish_time": time.time(),
                    "last_step": runtime["max_steps"] - 1,
                    "last_val_step": last_val_record["step"] if last_val_record is not None else None,
                    "best_score": best_score,
                },
            )
    except BaseException as exc:
        run_exception = exc
        if rank == 0 and "log_dir" in locals():
            write_status_json(
                log_dir / "run_status.json",
                {
                    "state": "failed",
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "start_time": run_start_time,
                    "fail_time": time.time(),
                    "last_step": last_train_record["step"] if "last_train_record" in locals() and last_train_record is not None else None,
                    "last_val_step": last_val_record["step"] if "last_val_record" in locals() and last_val_record is not None else None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        raise
    finally:
        cleanup_distributed(ddp_enabled)


if __name__ == "__main__":
    main()
