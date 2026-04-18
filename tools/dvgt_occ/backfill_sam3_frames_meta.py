#!/usr/bin/env python3
"""Backfill scene-level SAM3 frame metadata for valid/ignore supervision masks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.data.manifest import load_manifest
from dvgt_occ.data.stage_a_utils import binary_mask_to_rle, entry_output_dirs, manifest_scene_groups, rle_to_binary_mask


def _load_instances(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _build_frames_meta(records: list[dict], frame_ids_10hz: Iterable[int], view_id: int) -> list[dict]:
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    for record in records:
        by_frame[int(record['frame_id_10hz'])].append(record)
    frames_meta: list[dict] = []
    for scene_frame_index, frame_id in enumerate(frame_ids_10hz):
        frame_records = by_frame.get(int(frame_id), [])
        quality_masks = []
        rejected = 0
        for record in frame_records:
            if bool(record.get('quality_ok', False)) and (not bool(record.get('box_like_flag', False))):
                quality_masks.append(rle_to_binary_mask(record['rle']))
            else:
                rejected += 1
        exhaustive_ok = bool(quality_masks) and rejected == 0
        valid_region_rle = None
        if (not exhaustive_ok) and quality_masks:
            union = np.logical_or.reduce(quality_masks)
            valid_region_rle = binary_mask_to_rle(union)
        frames_meta.append(
            {
                'scene_frame_index': int(scene_frame_index),
                'frame_id_10hz': int(frame_id),
                'view_id': int(view_id),
                'exhaustive_ok': bool(exhaustive_ok),
                'valid_region_rle': valid_region_rle,
                'quality_ok_count': int(len(quality_masks)),
                'rejected_count': int(rejected),
            }
        )
    return frames_meta


def main() -> None:
    parser = argparse.ArgumentParser(description='Backfill SAM3 frames_meta.jsonl from scene-level instance assets.')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--scene-ids', nargs='*', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    output_root = (args.output_root if args.output_root is not None else args.manifest.parent).resolve()
    entries = load_manifest(args.manifest)
    if args.scene_ids:
        allowed = {str(scene_id) for scene_id in args.scene_ids}
        entries = [entry for entry in entries if str(entry['scene_id']) in allowed]
    scene_groups = manifest_scene_groups(entries)

    written = 0
    skipped = 0
    for _, entry in scene_groups:
        dirs = entry_output_dirs(entry, output_root)
        scene_meta_path = dirs['sam3_scene'] / 'scene_meta.json'
        if not scene_meta_path.exists():
            continue
        scene_meta = json.loads(scene_meta_path.read_text(encoding='utf-8'))
        frame_ids = scene_meta['frame_ids_10hz']
        for view_id in scene_meta['view_ids']:
            view_dir = dirs['sam3_scene'] / f'view_{int(view_id)}'
            out_path = view_dir / 'frames_meta.jsonl'
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            records = _load_instances(view_dir / 'instances.jsonl')
            frames_meta = _build_frames_meta(records, frame_ids, int(view_id))
            view_dir.mkdir(parents=True, exist_ok=True)
            with out_path.open('w', encoding='utf-8') as handle:
                for record in frames_meta:
                    handle.write(json.dumps(record, ensure_ascii=True) + '\n')
            written += 1
            exhaustive = sum(int(record['exhaustive_ok']) for record in frames_meta)
            print(f'[sam3_frame_meta] {entry["scene_id"]} view_{int(view_id)} frames={len(frames_meta)} exhaustive={exhaustive}')
    print(f'Wrote {written} frames_meta files; skipped {skipped}.')


if __name__ == '__main__':
    main()
