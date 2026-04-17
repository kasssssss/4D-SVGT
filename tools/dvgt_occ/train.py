#!/usr/bin/env python3
"""Stage-C/D training entry for DVGT-Occ."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import timedelta
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
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
    move_batch_to_device,
    reduce_metrics,
    resolve_loss_weights,
    stage_b_after_stability_weights,
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
        "grad_clip": float(train_cfg.get("grad_clip", 1.0)),
        "amp": bool(train_cfg.get("amp", True)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
        "warmup_iters": int(train_cfg.get("warmup_iters", 1500)),
        "min_lr": float(train_cfg.get("min_lr", 1e-6)),
        "stability_start_step": int(train_cfg.get("stability_start_step", 0)),
        "allow_tf32": bool(train_cfg.get("allow_tf32", True)),
        "log_dir": Path(args.log_dir or train_cfg.get("log_dir", "data/nuscenes_dvgt_v0/logs/train_stage")),
        "find_unused_parameters": bool(args.find_unused_parameters or ddp_cfg.get("find_unused_parameters", False)),
        "broadcast_buffers": bool(args.broadcast_buffers or ddp_cfg.get("broadcast_buffers", False)),
        "static_graph": bool(ddp_cfg.get("static_graph", False)),
        "ddp_timeout_sec": int(args.ddp_timeout_sec or ddp_cfg.get("timeout_sec", 1800)),
        "ddp_backend": str(args.ddp_backend or ddp_cfg.get("backend", "nccl")),
        "resume": Path(args.resume or train_cfg.get("resume", "")) if (args.resume or train_cfg.get("resume")) else None,
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


def build_loss_weights(cfg: Dict[str, object]) -> tuple[LossWeights, LossWeights]:
    weights_cfg = dict(cfg.get("loss_weights", {}))
    merged = DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS.to_dict()
    merged.update(weights_cfg.get("base", weights_cfg))
    base = LossWeights(**merged)

    after_cfg = weights_cfg.get("after_stability")
    after = stage_b_after_stability_weights()
    if after_cfg:
        after_merged = after.to_dict()
        after_merged.update(after_cfg)
        after = LossWeights(**after_merged)
    return base, after


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


def save_checkpoint(path: Path, step: int, raw_model: torch.nn.Module, optimizer: torch.optim.Optimizer, score: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"step": step, "model": raw_model.state_dict(), "optimizer": optimizer.state_dict()}
    if score is not None:
        payload["score"] = score
    torch.save(payload, path)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loss_builder: DVGTOccLossBuilder,
    loader: DataLoader,
    device: torch.device,
    ddp_enabled: bool,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    loss_totals: Dict[str, torch.Tensor] = {}
    denom = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        loss_total, loss_dict = loss_builder(outputs, batch)
        occ_logits = outputs["occ"].occ_logit
        occ_target = batch["occ_label"]
        dyn_logits = torch.logit(outputs["render"].render_alpha_dynamic.clamp(1e-4, 1.0 - 1e-4))
        dyn_target = batch["sam3_dyn_mask_full"].unsqueeze(3)
        metrics = {
            "val_loss_total": loss_total.detach(),
            "val_occ_iou": binary_iou_from_logits(occ_logits, occ_target),
            "val_dyn_mask_iou": binary_iou_from_logits(dyn_logits, dyn_target),
        }
        for key, value in binary_stats_from_logits(occ_logits, occ_target).items():
            metrics[f"val_occ_{key}"] = value
        for key, value in binary_stats_from_logits(dyn_logits, dyn_target).items():
            metrics[f"val_dyn_mask_{key}"] = value
        for key, value in loss_dict.items():
            metrics[f"val_loss_{key}"] = value.detach()
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
    try:
        model_cfg = build_model_config(cfg)
        set_seed(runtime["seed"] + rank)
        device = resolve_device(args, ddp_enabled, local_rank)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = runtime["allow_tf32"]
            torch.backends.cudnn.allow_tf32 = runtime["allow_tf32"]

        train_scene_ids, val_scene_ids = discover_scene_splits(runtime["manifest"], runtime["train_scene_ids"], runtime["val_scene_ids"])
        train_dataset = build_dataset(runtime["manifest"], runtime["output_root"], model_cfg, train_scene_ids, runtime["train_limit"])
        val_dataset = build_dataset(runtime["manifest"], runtime["output_root"], model_cfg, val_scene_ids, runtime["val_limit"]) if val_scene_ids else None
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

        base_weights, after_weights = build_loss_weights(cfg)
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
                "runtime": {key: str(value) if isinstance(value, Path) else value for key, value in runtime.items()},
                "model": asdict(model_cfg),
                "train_scene_ids": train_scene_ids,
                "val_scene_ids": val_scene_ids,
                "base_loss_weights": base_weights.to_dict(),
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

        model.train()
        train_iter = iter(train_loader)
        best_score = None
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
                after_stability=after_weights,
                stability_start_step=runtime["stability_start_step"],
            )
            adjust_learning_rate(optimizer, step, runtime["max_steps"], runtime["warmup_iters"], runtime["min_lr"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, enabled=autocast_enabled):
                outputs = model(batch)
                loss_total, loss_dict = loss_builder(outputs, batch)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), runtime["grad_clip"])
            optimizer.step()

            if rank == 0 and (step % runtime["log_interval"] == 0 or step == runtime["max_steps"] - 1):
                record = {
                    "step": step,
                    "loss_total": float(loss_total.detach().item()),
                    "lr_base": optimizer.param_groups[0]["lr"],
                    "scene_id": batch["scene_id"],
                    "clip_id": batch["clip_id"],
                    "loss_weights": loss_builder.weights.to_dict(),
                }
                record.update({f"loss_{key}": float(value.detach().item()) for key, value in loss_dict.items()})
                print(json.dumps(record, ensure_ascii=False))
                with (log_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if rank == 0 and runtime["save_interval"] > 0 and (step + 1) % runtime["save_interval"] == 0:
                save_checkpoint(log_dir / f"step_{step + 1:06d}.pt", step, raw_model, optimizer)

            if val_loader is not None and runtime["val_interval"] > 0 and (step + 1) % runtime["val_interval"] == 0:
                val_metrics = evaluate(model, loss_builder, val_loader, device, ddp_enabled, runtime["val_batches"])
                if rank == 0:
                    val_metrics["step"] = step
                    print(json.dumps(val_metrics, ensure_ascii=False))
                    with (log_dir / "val_metrics.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(val_metrics, ensure_ascii=False) + "\n")
                    score = val_metrics["val_occ_iou"] + val_metrics["val_dyn_mask_iou"]
                    if best_score is None or score > best_score:
                        best_score = score
                        save_checkpoint(log_dir / "best.pt", step, raw_model, optimizer, score=score)

        if rank == 0:
            save_checkpoint(log_dir / "last.pt", runtime["max_steps"] - 1, raw_model, optimizer, score=best_score)
    finally:
        cleanup_distributed(ddp_enabled)


if __name__ == "__main__":
    main()
