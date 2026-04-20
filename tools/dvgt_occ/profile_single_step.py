#!/usr/bin/env python3
"""Profile one DVGT-Occ training step on a single GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ import DVGTOccConfig
from dvgt_occ.data import DVGTOccClipDataset
from dvgt_occ.training import (
    DEFAULT_CACHE_KEYS,
    DEFAULT_SUPERVISION_KEYS,
    DVGTOccLossBuilder,
    collate_dvgt_occ_batch,
    move_batch_to_device,
)
from dvgt_occ.models import DVGTOccModel

try:
    from tools.dvgt_occ.train import build_loss_weights
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import build_loss_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one DVGT-Occ training step.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--clip-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    data_cfg = cfg["data"]
    model_cfg = DVGTOccConfig(**cfg["model"])
    device = torch.device(args.device)

    dataset = DVGTOccClipDataset(
        manifest_path=Path(data_cfg["manifest"]),
        root=Path(data_cfg["output_root"]),
        load_cache=True,
        load_supervision=True,
        cache_keys=DEFAULT_CACHE_KEYS,
        supervision_keys=DEFAULT_SUPERVISION_KEYS,
        load_scene_sam3_full=True,
        scene_ids=[args.scene_id] if args.scene_id else None,
        clip_ids=[args.clip_id] if args.clip_id else None,
        limit=1,
        projected_semantic_classes=model_cfg.projected_semantic_classes,
    )
    if len(dataset) == 0:
        raise SystemExit("No sample matched profiler selection.")

    batch = collate_dvgt_occ_batch([dataset[0]])
    base_weights, _, _ = build_loss_weights(cfg)
    batch["_active_loss_weights"] = base_weights.to_dict()
    batch["_profile_timing"] = True
    batch = move_batch_to_device(batch, device)

    model = DVGTOccModel(config=model_cfg).to(device)
    model.train()
    if hasattr(model, "configure_optional_modules"):
        model.configure_optional_modules(batch["_active_loss_weights"])
    loss_builder = DVGTOccLossBuilder(config=model_cfg, weights=base_weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    for _ in range(2):
        outputs = model(batch)
        outputs["step_hint"] = 0
        loss_total, _ = loss_builder(outputs, batch)
        loss_total.backward()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    outputs = model(batch)
    outputs["step_hint"] = 0
    loss_total, _ = loss_builder(outputs, batch)
    loss_total.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings_ms = outputs.get("timings_ms", {})
    ordered = sorted(timings_ms.items(), key=lambda kv: kv[1], reverse=True)
    summary_lines = ["Per-stage timings (ms)"]
    summary_lines.extend(f"{name:>24}: {value:8.3f}" for name, value in ordered)
    summary = "\n".join(summary_lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary, encoding="utf-8")
    meta = {
        "scene_id": batch["scene_id"],
        "clip_id": batch["clip_id"],
        "loss_total": float(loss_total.detach().item()),
        "timings_ms": timings_ms,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
