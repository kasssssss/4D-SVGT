#!/usr/bin/env python3
"""Quality-check DVGT cache by rerunning frozen DVGT on a clip and visualizing the point cloud."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dvgt_occ.config import DEFAULT_DVGT_OCC_CONFIG
from dvgt_occ.data.manifest import load_manifest
from cache_dvgt import _amp_dtype, _build_model, _load_images


def _select_entry(entries: list[dict], scene_id: str | None, clip_id: str | None, index: int) -> dict:
    if scene_id is None and clip_id is None:
        return entries[index]
    for entry in entries:
        if scene_id is not None and str(entry["scene_id"]) != str(scene_id):
            continue
        if clip_id is not None and str(entry["clip_id"]) != str(clip_id):
            continue
        return entry
    raise ValueError(f"Could not find manifest entry for scene_id={scene_id!r}, clip_id={clip_id!r}.")


def _load_cache(cache_dir: Path, config) -> dict[str, np.ndarray]:
    arrays = {
        "points": np.load(cache_dir / "points.npy", allow_pickle=False),
        "points_conf": np.load(cache_dir / "points_conf.npy", allow_pickle=False),
        "ego_pose": np.load(cache_dir / "ego_pose.npy", allow_pickle=False),
    }
    for layer in config.selected_layers:
        arrays[f"feat_l{layer}"] = np.load(cache_dir / f"feat_l{layer}.npy", allow_pickle=False)
    return arrays


def _compare_arrays(reference: np.ndarray, cached: np.ndarray) -> dict[str, float | list[int]]:
    diff = np.abs(reference.astype(np.float32) - cached.astype(np.float32))
    return {
        "shape": list(reference.shape),
        "dtype_reference": str(reference.dtype),
        "dtype_cached": str(cached.dtype),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff * diff))),
    }


def _flatten_points(points: np.ndarray, conf: np.ndarray, conf_percentile: float, max_depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf_cutoff = np.percentile(conf, conf_percentile)
    mask = conf >= conf_cutoff
    xyz = points[mask]
    conf_values = conf[mask]
    view_idx = np.broadcast_to(np.arange(points.shape[1])[None, :, None, None], points.shape[:4])[mask]
    depth = np.linalg.norm(xyz, axis=-1)
    if max_depth > 0:
        keep = depth <= max_depth
        xyz = xyz[keep]
        conf_values = conf_values[keep]
        view_idx = view_idx[keep]
    return xyz, conf_values, view_idx


def _plot_point_cloud(points: np.ndarray, conf: np.ndarray, view_idx: np.ndarray, out_path: Path, title: str) -> None:
    if points.shape[0] == 0:
        raise RuntimeError("No points survived the QC filtering; try lowering the confidence percentile or max depth filter.")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    bev = axes[0].scatter(points[:, 0], points[:, 1], c=conf, s=0.6, cmap="viridis", alpha=0.8, linewidths=0.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title(f"{title} BEV colored by confidence")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")
    fig.colorbar(bev, ax=axes[0], shrink=0.8)

    palette = np.array(
        [
            [220, 20, 60],
            [30, 144, 255],
            [34, 139, 34],
            [255, 140, 0],
            [148, 0, 211],
            [0, 206, 209],
        ],
        dtype=np.float32,
    ) / 255.0
    colors = palette[np.mod(view_idx.astype(np.int64), len(palette))]
    axes[1].scatter(points[:, 0], points[:, 2], c=colors, s=0.6, alpha=0.8, linewidths=0.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_title(f"{title} XZ colored by view id")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="QC frozen DVGT cache by rerunning the original decode head.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", "--root", dest="output_root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/dvgt2.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--clip-id", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--conf-percentile", type=float, default=75.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--qc-dir", type=Path, default=None)
    args = parser.parse_args()

    output_root = (args.output_root if args.output_root is not None else args.manifest.parent).resolve()
    config = DEFAULT_DVGT_OCC_CONFIG
    entries = load_manifest(args.manifest)
    entry = _select_entry(entries, scene_id=args.scene_id, clip_id=args.clip_id, index=args.index)
    cache_dir = output_root / entry["dvgt_cache_dir"]
    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing cache directory: {cache_dir}")

    qc_dir = (
        args.qc_dir.resolve()
        if args.qc_dir is not None
        else (output_root / "qc" / "dvgt_cache" / f"scene_{entry['scene_id']}" / entry["clip_id"]).resolve()
    )
    qc_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    amp_dtype = _amp_dtype(args.amp_dtype)

    print(f"[qc] scene={entry['scene_id']} clip={entry['clip_id']} device={device}")
    images = _load_images(entry, output_root, size_hw=(config.image_height, config.image_width)).to(device)
    model = _build_model(args.checkpoint.resolve(), device)
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

    rerun = {
        "points": points.squeeze(0).detach().cpu().float().numpy().astype(np.float32),
        "points_conf": points_conf.squeeze(0).detach().cpu().float().numpy().astype(np.float32),
    }
    for layer in config.selected_layers:
        rerun[f"feat_l{layer}"] = aggregated_tokens_list[layer].squeeze(0).detach().cpu().to(torch.float16).numpy()

    cached = _load_cache(cache_dir, config)
    summary = {
        "scene_id": str(entry["scene_id"]),
        "clip_id": str(entry["clip_id"]),
        "frame_ids_10hz": list(entry["frame_ids_10hz"]),
        "view_ids": list(entry["view_ids"]),
        "patch_start_idx_rerun": int(patch_start_idx),
        "comparisons": {
            "points": _compare_arrays(rerun["points"], cached["points"]),
            "points_conf": _compare_arrays(rerun["points_conf"], cached["points_conf"]),
        },
    }
    for layer in config.selected_layers:
        summary["comparisons"][f"feat_l{layer}"] = _compare_arrays(rerun[f"feat_l{layer}"], cached[f"feat_l{layer}"])

    xyz, conf_values, view_idx = _flatten_points(
        cached["points"],
        cached["points_conf"],
        conf_percentile=float(args.conf_percentile),
        max_depth=float(args.max_depth),
    )
    summary["point_cloud_stats"] = {
        "filtered_points": int(xyz.shape[0]),
        "conf_percentile": float(args.conf_percentile),
        "max_depth": float(args.max_depth),
        "xyz_min": xyz.min(axis=0).astype(float).tolist(),
        "xyz_max": xyz.max(axis=0).astype(float).tolist(),
        "xyz_mean": xyz.mean(axis=0).astype(float).tolist(),
    }

    _plot_point_cloud(
        xyz,
        conf_values,
        view_idx,
        qc_dir / "dvgt_cache_point_cloud.png",
        title=f"DVGT cache scene {entry['scene_id']} {entry['clip_id']}",
    )
    (qc_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[qc] wrote {qc_dir / 'qc_summary.json'}")
    print(f"[qc] wrote {qc_dir / 'dvgt_cache_point_cloud.png'}")


if __name__ == "__main__":
    main()
