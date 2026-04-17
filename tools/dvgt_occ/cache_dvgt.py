#!/usr/bin/env python3
"""Generate full frozen-DVGT cache for Stage-A clips."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.config import DEFAULT_DVGT_OCC_CONFIG
from dvgt_occ.data.manifest import load_manifest, resolve_manifest_path
from dvgt_occ.data.stage_a_utils import (
    distributed_shard_spec,
    entry_output_dirs,
    filter_manifest_entries,
    relative_ego_poses,
    required_files_exist,
    shard_items,
)


OUTPUT_FILES = (
    "points.npy",
    "points_conf.npy",
    "ego_pose.npy",
    "feat_l4.npy",
    "feat_l11.npy",
    "feat_l17.npy",
    "feat_l23.npy",
)

DEFAULT_DVGT_GT_SCALE_FACTOR = 0.1


def _load_images(entry: dict, output_root: Path, size_hw: tuple[int, int]):
    import torch
    from PIL import Image

    view_ids = [str(view_id) for view_id in entry["view_ids"]]
    frames = []
    for frame_idx in range(len(entry["frame_ids_10hz"])):
        views = []
        for view_id in view_ids:
            path = resolve_manifest_path(output_root, entry["image_paths"][view_id][frame_idx])
            image = Image.open(path).convert("RGB")
            image = image.resize((size_hw[1], size_hw[0]), Image.Resampling.BICUBIC)
            arr = np.asarray(image, dtype=np.float32) / 255.0
            views.append(torch.from_numpy(arr).permute(2, 0, 1))
        frames.append(torch.stack(views, dim=0))
    return torch.stack(frames, dim=0).unsqueeze(0).contiguous()


def _amp_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported amp dtype: {name}")


def _build_model(checkpoint: Path, device: str):
    import torch
    _install_import_shims()
    from dvgt.models.architectures.dvgt2 import DVGT2

    original_hub_load = torch.hub.load

    def _local_hub_load(repo_or_dir, model_name, *args, **kwargs):
        if str(repo_or_dir) == "third_party/dinov3" and "weights" not in kwargs:
            kwargs.setdefault("pretrained", False)
        return original_hub_load(repo_or_dir, model_name, *args, **kwargs)

    torch.hub.load = _local_hub_load
    try:
        model = DVGT2(
            dino_v3_weight_path=None,
            future_frame_window=8,
            relative_pose_window=1,
            use_causal_mask=True,
            ego_pose_head_conf=None,
        )
    finally:
        torch.hub.load = original_hub_load
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("ego_pose_head.")]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys outside ego_pose_head: {unexpected[:20]}")
    if missing:
        print(f"[cache_dvgt] Missing keys while loading frozen DVGT: {missing[:20]}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _install_import_shims() -> None:
    """Provide tiny import shims for optional DVGT deps unused by this tool."""

    if "iopath.common.file_io" not in sys.modules:
        iopath_mod = types.ModuleType("iopath")
        common_mod = types.ModuleType("iopath.common")
        file_io_mod = types.ModuleType("iopath.common.file_io")

        class _PathManager:
            @staticmethod
            def open(path, mode="r", *args, **kwargs):
                return open(path, mode, *args, **kwargs)

            @staticmethod
            def exists(path):
                return Path(path).exists()

            @staticmethod
            def isdir(path):
                return Path(path).is_dir()

        file_io_mod.g_pathmgr = _PathManager()
        sys.modules.setdefault("iopath", iopath_mod)
        sys.modules.setdefault("iopath.common", common_mod)
        sys.modules.setdefault("iopath.common.file_io", file_io_mod)
    if "hydra.utils" not in sys.modules:
        hydra_mod = types.ModuleType("hydra")
        hydra_utils_mod = types.ModuleType("hydra.utils")

        def _instantiate(config, *args, **kwargs):
            if config is None:
                return None
            raise RuntimeError("Hydra instantiate shim only supports None in cache_dvgt.py")

        hydra_utils_mod.instantiate = _instantiate
        sys.modules.setdefault("hydra", hydra_mod)
        sys.modules.setdefault("hydra.utils", hydra_utils_mod)
    if "torchmetrics" not in sys.modules:
        import torch.nn as nn

        torchmetrics_mod = types.ModuleType("torchmetrics")

        class _Metric(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def update(self, *args, **kwargs):
                return None

            def compute(self):
                return None

        torchmetrics_mod.Metric = _Metric
        sys.modules.setdefault("torchmetrics", torchmetrics_mod)
    if "termcolor" not in sys.modules:
        termcolor_mod = types.ModuleType("termcolor")
        termcolor_mod.colored = lambda text, *args, **kwargs: text
        sys.modules.setdefault("termcolor", termcolor_mod)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen DVGT full cache for manifest clips.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", "--root", dest="output_root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/dvgt2.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--gt-scale-factor", type=float, default=DEFAULT_DVGT_GT_SCALE_FACTOR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--clip-ids", nargs="*", default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--no-rank-env", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch

    output_root = (args.output_root if args.output_root is not None else args.manifest.parent).resolve()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing DVGT checkpoint: {checkpoint}")
    entries = load_manifest(args.manifest)
    entries = filter_manifest_entries(entries, scene_ids=args.scene_ids, clip_ids=args.clip_ids)
    if args.limit is not None:
        entries = entries[: args.limit]
    shard_index, num_shards = distributed_shard_spec(
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        use_env=not args.no_rank_env,
    )
    entries = shard_items(entries, shard_index=shard_index, num_shards=num_shards)
    model = _build_model(checkpoint, device)
    config = DEFAULT_DVGT_OCC_CONFIG
    amp_dtype = _amp_dtype(args.amp_dtype)
    points_metric_scale = 1.0 / float(args.gt_scale_factor) if float(args.gt_scale_factor) != 0.0 else 1.0
    built = 0
    skipped = 0
    for entry in entries:
        dirs = entry_output_dirs(entry, output_root)
        cache_dir = dirs["dvgt_cache"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite and required_files_exist(cache_dir, OUTPUT_FILES):
            skipped += 1
            continue

        images = _load_images(entry, output_root, size_hw=(config.image_height, config.image_width)).to(device)
        with torch.no_grad(), torch.amp.autocast(
            device_type=torch.device(device).type,
            enabled=(amp_dtype != torch.float32 and torch.device(device).type == "cuda"),
            dtype=amp_dtype,
        ):
            aggregated_tokens_list, patch_start_idx, _ = model.aggregator(images)
            points, points_conf = model.point_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=model.frames_chunk_size,
            )

        points_metric = points.squeeze(0).detach().cpu().float().numpy().astype(np.float32) * np.float32(points_metric_scale)
        np.save(cache_dir / "points.npy", points_metric)
        np.save(cache_dir / "points_conf.npy", points_conf.squeeze(0).detach().cpu().float().numpy().astype(np.float32))
        np.save(cache_dir / "ego_pose.npy", relative_ego_poses(dirs["source_scene"], entry["frame_ids_10hz"]))
        np.save(cache_dir / "patch_start_idx.npy", np.array([patch_start_idx], dtype=np.int64))
        for layer in config.selected_layers:
            tokens = aggregated_tokens_list[layer].squeeze(0).detach().cpu().to(torch.float16).numpy()
            np.save(cache_dir / f"feat_l{layer}.npy", tokens)
        (cache_dir / "cache_meta.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "selected_layers": list(config.selected_layers),
                    "patch_start_idx": int(patch_start_idx),
                    "image_shape": [config.image_height, config.image_width],
                    "token_dtype": "float16",
                    "geometry_dtype": "float32",
                    "gt_scale_factor": float(args.gt_scale_factor),
                    "points_metric_scale": float(points_metric_scale),
                    "cache_points_are_metric": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        del images, aggregated_tokens_list, points, points_conf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        built += 1
        print(f"[cache][{shard_index}/{num_shards}] {entry['scene_id']}/{entry['clip_id']} saved to {cache_dir}")
    print(f"Generated DVGT cache for {built} clips; skipped {skipped}; shard {shard_index}/{num_shards}.")


if __name__ == "__main__":
    main()
