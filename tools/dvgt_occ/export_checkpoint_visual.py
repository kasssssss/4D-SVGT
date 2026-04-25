#!/usr/bin/env python3
"""Export a single qualitative visualization from a saved checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dvgt_occ.data import DVGTOccClipDataset
from dvgt_occ.models import DVGTOccModel
from dvgt_occ.training import (
    DEFAULT_CACHE_KEYS,
    DEFAULT_SUPERVISION_KEYS,
    collate_dvgt_occ_batch,
    move_batch_to_device,
    save_training_visualization,
)
from train import build_model_config, build_runtime_config, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a single DVGT-Occ checkpoint visualization.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--view-idx", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = build_runtime_config(argparse.Namespace(
        manifest=None,
        output_root=None,
        train_scene_ids=None,
        val_scene_ids=None,
        train_limit=None,
        val_limit=None,
        max_steps=None,
        batch_size=1,
        num_workers=0,
        seed=None,
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
    model_cfg = build_model_config(cfg)
    device = torch.device(args.device)

    if args.split == "train":
        scene_ids = runtime["train_scene_ids"]
        limit = runtime["train_limit"]
    else:
        scene_ids = runtime["val_scene_ids"]
        limit = runtime["val_limit"]

    dataset = DVGTOccClipDataset(
        manifest_path=runtime["manifest"],
        root=runtime["output_root"],
        load_cache=True,
        load_supervision=True,
        projected_semantic_classes=model_cfg.projected_semantic_classes,
        cache_keys=DEFAULT_CACHE_KEYS,
        supervision_keys=DEFAULT_SUPERVISION_KEYS,
        scene_ids=scene_ids,
        limit=limit,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split '{args.split}' is empty.")
    sample = dataset[min(max(args.index, 0), len(dataset) - 1)]
    batch = collate_dvgt_occ_batch([sample])
    batch = move_batch_to_device(batch, device)

    model = DVGTOccModel(config=model_cfg).to(device)
    if hasattr(model, "set_train_target"):
        model.set_train_target("gs_only")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    batch["_train_mode"] = "v1-stable"

    with torch.no_grad():
        outputs = model(batch)

    step = int(checkpoint.get("step", 0))
    epoch = int(checkpoint.get("epoch", 0))
    save_training_visualization(
        batch,
        outputs,
        model_cfg,
        args.out,
        step=step,
        epoch=epoch,
        sample_idx=0,
        frame_idx=args.frame_idx,
        view_idx=args.view_idx,
    )
    print(args.out)


if __name__ == "__main__":
    main()
