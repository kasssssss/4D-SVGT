#!/usr/bin/env python3
"""Build clip-level SAM3 supervision from scene-level primary assets."""

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
from dvgt_occ.data.stage_a_utils import (
    binary_mask_boundary,
    build_track_arrays,
    compute_track_speed_scores,
    distributed_shard_spec,
    entry_output_dirs,
    filter_manifest_entries,
    resize_mask_float,
    rle_to_binary_mask,
    shard_items,
)


OUTPUT_FILES = (
    "dyn_soft_mask_1_4.npy",
    "gs_local_inst_id_1_8.npy",
    "gs_global_track_id_1_8.npy",
    "inst_boundary_weight_1_8.npy",
    "inst_area_ratio_1_8.npy",
    "inst_quality_mask_1_8.npy",
)


def _record_accepted_for_training(record: dict) -> bool:
    return bool(record.get("quality_ok", False)) and (not bool(record.get("box_like_flag", False)))


def _load_scene_records(scene_dir: Path, view_ids: Iterable[int]) -> Dict[int, Dict[int, List[dict]]]:
    records: Dict[int, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for view_id in view_ids:
        path = scene_dir / f"view_{view_id}" / "instances.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                records[int(record["frame_id_10hz"])][int(record["view_id"])].append(record)
    return records


def _fallback_dynamic_score(record: dict) -> float:
    if record["class_name"] in {"pedestrian", "bicycle", "motorcycle"}:
        return 0.8
    return 0.5


def _clip_record_index(entry: dict, scene_records: Dict[int, Dict[int, List[dict]]]) -> dict:
    records = {}
    refs = {}
    for t_idx, frame_id in enumerate(entry["frame_ids_10hz"]):
        for v_idx, view_id in enumerate(entry["view_ids"]):
            key = f"{t_idx}_{v_idx}"
            selected = scene_records.get(int(frame_id), {}).get(int(view_id), [])
            refs[key] = [record["record_id"] for record in selected]
            for record in selected:
                records[record["record_id"]] = {
                    "scene_frame_index": int(record["scene_frame_index"]),
                    "frame_id_10hz": int(record["frame_id_10hz"]),
                    "view_id": int(record["view_id"]),
                    "local_mask_id": int(record["local_mask_id"]),
                    "global_weak_id": int(record.get("global_weak_id", -1)),
                    "class_name": record["class_name"],
                    "score": float(record["score"]),
                    "quality_ok": bool(record["quality_ok"]),
                    "box_like_flag": bool(record["box_like_flag"]),
                    "quality_reason": record.get("quality_reason", "unknown"),
                    "refined": bool(record["refined"]),
                    "visible_area": int(record["visible_area"]),
                    "bbox_mask_tight": list(record["bbox_mask_tight"]),
                    "match_score": float(record.get("match_score", 0.0)),
                    "matched_track_id": int(record.get("matched_track_id", -1)),
                    "matched_proj_box_xyxy": record.get("matched_proj_box_xyxy"),
                    "matched_proj_box_xywh": record.get("matched_proj_box_xywh"),
                }
    return {
        "scene_id": entry["scene_id"],
        "clip_id": entry["clip_id"],
        "frame_ids_10hz": entry["frame_ids_10hz"],
        "view_ids": entry["view_ids"],
        "records": records,
        "records_by_frame_view": refs,
    }


def _derive_frame_supervision(records: List[dict], dynamic_scores: Dict[int, np.ndarray], t_idx: int) -> tuple[np.ndarray, ...]:
    dyn_soft = np.zeros((56, 112), dtype=np.float32)
    local_map = np.full((28, 56), -1, dtype=np.int32)
    global_map = np.full((28, 56), -1, dtype=np.int64)
    area_ratio = np.zeros((28, 56), dtype=np.float32)
    quality_mask = np.zeros((28, 56), dtype=np.float32)
    boundary_weight = np.zeros((28, 56), dtype=np.float32)
    local_ratio = np.zeros((28, 56), dtype=np.float32)
    global_ratio = np.zeros((28, 56), dtype=np.float32)

    for record in records:
        accepted = _record_accepted_for_training(record)
        mask = rle_to_binary_mask(record["rle"])
        ratio_1_4 = resize_mask_float(mask, (56, 112))
        ratio_1_8 = resize_mask_float(mask, (28, 56))
        boundary_1_8 = resize_mask_float(binary_mask_boundary(mask), (28, 56))
        if accepted:
            boundary_weight = np.maximum(boundary_weight, boundary_1_8)
            dyn_score = dynamic_scores.get(int(record.get("global_weak_id", -1)))
            if dyn_score is not None and t_idx < len(dyn_score):
                score = float(dyn_score[t_idx])
            else:
                score = _fallback_dynamic_score(record)
            dyn_soft = np.maximum(dyn_soft, ratio_1_4 * score)
            local_update = ratio_1_8 > local_ratio
            local_map[local_update] = int(record["local_mask_id"])
            local_ratio[local_update] = ratio_1_8[local_update]
            area_ratio[local_update] = ratio_1_8[local_update]
            quality_mask[local_update] = 1.0

            if int(record.get("global_weak_id", -1)) >= 0:
                global_update = ratio_1_8 > global_ratio
                global_map[global_update] = int(record["global_weak_id"])
                global_ratio[global_update] = ratio_1_8[global_update]

    return dyn_soft, local_map, global_map, boundary_weight, area_ratio, quality_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clip-level SAM3 supervision from scene assets.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
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

    scene_records_cache: Dict[str, Dict[int, Dict[int, List[dict]]]] = {}
    dynamic_score_cache: Dict[str, Dict[int, np.ndarray]] = {}

    built = 0
    skipped = 0
    for entry in entries:
        dirs = entry_output_dirs(entry, output_root)
        sup_dir = dirs["supervision"]
        sup_dir.mkdir(parents=True, exist_ok=True)
        clip_index_path = dirs["sam3_clip_index"]
        if clip_index_path.exists() and not args.overwrite and all((sup_dir / name).exists() for name in OUTPUT_FILES):
            skipped += 1
            continue

        scene_key = str(dirs["sam3_scene"])
        if scene_key not in scene_records_cache:
            scene_records_cache[scene_key] = _load_scene_records(dirs["sam3_scene"], entry["view_ids"])
        if scene_key not in dynamic_score_cache:
            scene_frame_ids = entry.get("scene_frame_ids_10hz", entry["frame_ids_10hz"])
            track_arrays = build_track_arrays(dirs["source_scene"], scene_frame_ids)
            dynamic_score_cache[scene_key] = compute_track_speed_scores(track_arrays, dt_seconds=0.1)
        scene_records = scene_records_cache[scene_key]
        dynamic_scores = dynamic_score_cache[scene_key]
        scene_frame_to_index = {int(frame_id): idx for idx, frame_id in enumerate(entry.get("scene_frame_ids_10hz", entry["frame_ids_10hz"]))}

        dyn_soft_all = []
        local_all = []
        global_all = []
        boundary_all = []
        area_all = []
        quality_all = []
        for t_idx, frame_id in enumerate(entry["frame_ids_10hz"]):
            dyn_views = []
            local_views = []
            global_views = []
            boundary_views = []
            area_views = []
            quality_views = []
            for view_id in entry["view_ids"]:
                frame_records = scene_records.get(int(frame_id), {}).get(int(view_id), [])
                dyn_soft, local_map, global_map, boundary_weight, area_ratio, quality_mask = _derive_frame_supervision(
                    frame_records,
                    dynamic_scores,
                    t_idx=scene_frame_to_index[int(frame_id)],
                )
                dyn_views.append(dyn_soft)
                local_views.append(local_map)
                global_views.append(global_map)
                boundary_views.append(boundary_weight)
                area_views.append(area_ratio)
                quality_views.append(quality_mask)
            dyn_soft_all.append(np.stack(dyn_views, axis=0))
            local_all.append(np.stack(local_views, axis=0))
            global_all.append(np.stack(global_views, axis=0))
            boundary_all.append(np.stack(boundary_views, axis=0))
            area_all.append(np.stack(area_views, axis=0))
            quality_all.append(np.stack(quality_views, axis=0))

        np.save(sup_dir / "dyn_soft_mask_1_4.npy", np.stack(dyn_soft_all, axis=0).astype(np.float32))
        np.save(sup_dir / "gs_local_inst_id_1_8.npy", np.stack(local_all, axis=0).astype(np.int32))
        np.save(sup_dir / "gs_global_track_id_1_8.npy", np.stack(global_all, axis=0).astype(np.int64))
        np.save(sup_dir / "inst_boundary_weight_1_8.npy", np.stack(boundary_all, axis=0).astype(np.float32))
        np.save(sup_dir / "inst_area_ratio_1_8.npy", np.stack(area_all, axis=0).astype(np.float32))
        np.save(sup_dir / "inst_quality_mask_1_8.npy", np.stack(quality_all, axis=0).astype(np.float32))

        clip_index = _clip_record_index(entry, scene_records)
        clip_index["sam3_scene_dir"] = entry.get("sam3_scene_dir")
        clip_index_path.write_text(json.dumps(clip_index, indent=2), encoding="utf-8")
        built += 1
        quality_ok_count = sum(1 for record in clip_index["records"].values() if bool(record.get("quality_ok", False)))
        accepted_count = sum(1 for record in clip_index["records"].values() if _record_accepted_for_training(record))
        print(
            f"[sam3_clip][{shard_index}/{num_shards}] {entry['scene_id']}/{entry['clip_id']} "
            f"records={len(clip_index['records'])} quality_ok={quality_ok_count} accepted={accepted_count}"
        )

    print(f"Built SAM3 clip supervision for {built} clips; skipped {skipped}; shard {shard_index}/{num_shards}.")


if __name__ == "__main__":
    main()
