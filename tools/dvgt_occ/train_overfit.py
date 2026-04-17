#!/usr/bin/env python3
"""Small-sample overfit trainer for the DVGT-Occ stage-B scaffold."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable

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
from dvgt_occ.models import DVGTOccModel
from dvgt_occ.training import (
    DEFAULT_CACHE_KEYS,
    DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS,
    DEFAULT_SUPERVISION_KEYS,
    DVGTOccLossBuilder,
    LossWeights,
    collate_dvgt_occ_batch,
    move_batch_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-B overfit training for DVGT-Occ.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dvgt_occ/train_overfit_v1.yaml"),
        help="YAML config path.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--clip-ids", nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
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


def init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
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
        "limit": args.limit if args.limit is not None else data_cfg.get("limit"),
        "scene_ids": args.scene_ids if args.scene_ids is not None else data_cfg.get("scene_ids"),
        "clip_ids": args.clip_ids if args.clip_ids is not None else data_cfg.get("clip_ids"),
        "batch_size": int(args.batch_size or train_cfg.get("batch_size", 1)),
        "num_workers": int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)),
        "max_steps": int(args.max_steps or train_cfg.get("max_steps", 20)),
        "seed": int(args.seed or train_cfg.get("seed", 42)),
        "log_interval": int(train_cfg.get("log_interval", 1)),
        "save_interval": int(train_cfg.get("save_interval", 0)),
        "grad_clip": float(train_cfg.get("grad_clip", 1.0)),
        "amp": bool(train_cfg.get("amp", True)),
        "warmup_iters": int(train_cfg.get("warmup_iters", 1500)),
        "min_lr": float(train_cfg.get("min_lr", 1e-6)),
        "log_dir": Path(args.log_dir or train_cfg.get("log_dir", "data/nuscenes_dvgt_v0/logs/overfit_stage_b")),
        "find_unused_parameters": bool(ddp_cfg.get("find_unused_parameters", False)),
        "static_graph": bool(ddp_cfg.get("static_graph", True)),
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

    param_groups = [group for group in groups.values() if group["params"]]
    return torch.optim.AdamW(param_groups, betas=betas, weight_decay=weight_decay)


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


def build_loss_weights(cfg: Dict[str, object]) -> LossWeights:
    weights_cfg = dict(cfg.get("loss_weights", {}))
    merged = DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS.to_dict()
    merged.update(weights_cfg)
    return LossWeights(**merged)


def reduce_scalar(value: torch.Tensor, enabled: bool) -> float:
    scalar = value.detach()
    if enabled:
        dist.all_reduce(scalar, op=dist.ReduceOp.AVG)
    return float(scalar.item())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ddp_enabled, rank, world_size, local_rank = init_distributed()
    try:
        runtime = build_runtime_config(args, cfg)
        model_cfg = build_model_config(cfg)
        set_seed(runtime["seed"] + rank)
        device = resolve_device(args, ddp_enabled, local_rank)

        dataset = DVGTOccClipDataset(
            manifest_path=runtime["manifest"],
            root=runtime["output_root"],
            load_cache=True,
            load_supervision=True,
            load_scene_sam3_full=True,
            full_res_size=(model_cfg.image_height, model_cfg.image_width),
            projected_semantic_classes=model_cfg.projected_semantic_classes,
            cache_keys=DEFAULT_CACHE_KEYS,
            supervision_keys=DEFAULT_SUPERVISION_KEYS,
            scene_ids=runtime["scene_ids"],
            clip_ids=runtime["clip_ids"],
            limit=runtime["limit"],
        )
        if len(dataset) == 0:
            raise RuntimeError("No clips selected for overfit training.")

        sampler = None
        if ddp_enabled:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        dataloader = DataLoader(
            dataset,
            batch_size=runtime["batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=runtime["num_workers"],
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_dvgt_occ_batch,
        )
        data_iter = cycle(dataloader)

        model = DVGTOccModel(config=model_cfg).to(device)
        if ddp_enabled:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=runtime["find_unused_parameters"],
                static_graph=runtime["static_graph"],
            )
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model

        loss_builder = DVGTOccLossBuilder(config=model_cfg, weights=build_loss_weights(cfg)).to(device)
        optimizer = build_optimizer(raw_model, cfg)
        autocast_enabled = bool(runtime["amp"] and device.type == "cuda")

        log_dir = runtime["log_dir"]
        if rank == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            config_snapshot = {
                "runtime": {key: str(value) if isinstance(value, Path) else value for key, value in runtime.items()},
                "model": asdict(model_cfg),
                "loss_weights": build_loss_weights(cfg).to_dict(),
            }
            (log_dir / "train_overfit_config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

        model.train()
        for step in range(runtime["max_steps"]):
            if sampler is not None and step % max(len(dataloader), 1) == 0:
                sampler.set_epoch(step)
            batch = move_batch_to_device(next(data_iter), device)
            adjust_learning_rate(
                optimizer,
                step=step,
                max_steps=runtime["max_steps"],
                warmup_iters=runtime["warmup_iters"],
                min_lr=runtime["min_lr"],
            )
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
                    "loss_total": reduce_scalar(loss_total, ddp_enabled),
                    "lr_base": optimizer.param_groups[0]["lr"],
                    "scene_id": batch["scene_id"],
                    "clip_id": batch["clip_id"],
                }
                for key, value in loss_dict.items():
                    record[f"loss_{key}"] = reduce_scalar(value, ddp_enabled)
                print(json.dumps(record, ensure_ascii=False))
                with (log_dir / "train_overfit_metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if rank == 0 and runtime["save_interval"] > 0 and (step + 1) % runtime["save_interval"] == 0:
                ckpt = {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(ckpt, log_dir / f"step_{step + 1:06d}.pt")

        if rank == 0:
            torch.save(
                {
                    "step": runtime["max_steps"] - 1,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                log_dir / "last.pt",
            )
    finally:
        cleanup_distributed(ddp_enabled)


if __name__ == "__main__":
    main()
