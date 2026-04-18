#!/usr/bin/env python3
"""Rewrite legacy DVGT cache points.npy into metric scale in-place."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.data.manifest import load_manifest, resolve_manifest_path
from dvgt_occ.data.stage_a_utils import distributed_shard_spec, filter_manifest_entries, shard_items


def load_meta(cache_dir: Path) -> dict:
    meta_path = cache_dir / "cache_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def save_meta(cache_dir: Path, meta: dict) -> None:
    meta_path = cache_dir / "cache_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def rewrite_points_file(points_path: Path, scale: float) -> None:
    points = np.load(points_path, allow_pickle=False).astype(np.float32)
    points = (points * np.float32(scale)).astype(np.float32)
    tmp_path = points_path.with_suffix(".tmp.npy")
    np.save(tmp_path, points)
    tmp_path.replace(points_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite legacy DVGT cached points.npy to metric scale.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", "--root", dest="output_root", type=Path, default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--clip-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gt-scale-factor", type=float, default=0.1)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--no-rank-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = (args.output_root if args.output_root is not None else args.manifest.parent).resolve()
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

    metric_scale = 1.0 / float(args.gt_scale_factor)
    rewritten = 0
    skipped = 0

    for entry in entries:
        cache_dir = resolve_manifest_path(output_root, entry["dvgt_cache_dir"])
        points_path = cache_dir / "points.npy"
        if not points_path.exists():
            print(f"[rewrite][skip] missing points: {entry['scene_id']}/{entry['clip_id']}")
            skipped += 1
            continue

        meta = load_meta(cache_dir)
        already_metric = bool(meta.get("cache_points_are_metric", False))
        existing_scale = float(meta.get("points_metric_scale", 1.0)) if "points_metric_scale" in meta else None
        if already_metric and not args.force:
            if existing_scale is None or abs(existing_scale - 1.0) > 1e-8:
                meta["gt_scale_factor"] = float(args.gt_scale_factor)
                meta["points_metric_scale"] = 1.0
                meta["cache_points_are_metric"] = True
                save_meta(cache_dir, meta)
            skipped += 1
            continue
        if existing_scale is not None and abs(existing_scale - metric_scale) < 1e-8 and not args.force:
            meta["cache_points_are_metric"] = True
            save_meta(cache_dir, meta)
            skipped += 1
            continue

        rewrite_points_file(points_path, metric_scale)
        meta["gt_scale_factor"] = float(args.gt_scale_factor)
        meta["points_metric_scale"] = 1.0
        meta["cache_points_are_metric"] = True
        save_meta(cache_dir, meta)
        rewritten += 1
        print(f"[rewrite][{shard_index}/{num_shards}] {entry['scene_id']}/{entry['clip_id']} -> metric x{metric_scale:g}")

    print(f"Rewrote {rewritten} clips; skipped {skipped}; shard {shard_index}/{num_shards}.")


if __name__ == "__main__":
    main()
