#!/usr/bin/env python3
"""Build 2Hz track and streaming-memory supervision arrays."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dvgt_occ.data.manifest import load_manifest
from dvgt_occ.data.stage_a_utils import (
    build_track_arrays,
    distributed_shard_spec,
    entry_output_dirs,
    filter_manifest_entries,
    required_files_exist,
    shard_items,
)


OUTPUT_FILES = (
    "track_boxes_3d.npy",
    "track_cls.npy",
    "track_id.npy",
    "track_visible.npy",
    "track_birth.npy",
    "track_death.npy",
    "track_delta_box.npy",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build track boxes, birth/death, and delta targets.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--clip-ids", nargs="*", default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--no-rank-env", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
    built = 0
    skipped = 0
    for entry in entries:
        dirs = entry_output_dirs(entry, output_root)
        sup_dir = dirs["supervision"]
        sup_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite and required_files_exist(sup_dir, OUTPUT_FILES):
            skipped += 1
            continue
        arrays = build_track_arrays(
            scene_dir=dirs["source_scene"],
            frame_ids=entry["frame_ids_10hz"],
            max_objects=args.max_objects,
        )
        for key, value in arrays.items():
            np.save(sup_dir / f"{key}.npy", value)
        built += 1
        print(
            f"[track][{shard_index}/{num_shards}] {entry['scene_id']}/{entry['clip_id']} "
            f"T={arrays['track_boxes_3d'].shape[0]} N={arrays['track_boxes_3d'].shape[1]}"
        )
    print(f"Built track-memory targets for {built} clips; skipped {skipped}; shard {shard_index}/{num_shards}.")


if __name__ == "__main__":
    main()
