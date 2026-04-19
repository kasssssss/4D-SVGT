#!/usr/bin/env python3
"""Audit cached DVGT-Occ supervision assets for shape/value consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.config import DEFAULT_DVGT_OCC_CONFIG
from dvgt_occ.data import DVGTOccClipDataset
from dvgt_occ.data.manifest import load_manifest, resolve_manifest_path
from dvgt_occ.data.stage_a_utils import resize_mask_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage-A supervision assets.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-clips", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = max(1, min(length, count))
    if count == 1:
        return [0]
    return sorted({int(round(i * (length - 1) / float(count - 1))) for i in range(count)})


def to_shape(value: np.ndarray | None) -> list[int] | None:
    return None if value is None else [int(x) for x in value.shape]


def audit_scene_level_records(entries: list[dict], root: Path) -> dict[str, Any]:
    scene_dirs = {}
    for entry in entries:
        scene_id = str(entry["scene_id"])
        scene_dirs.setdefault(scene_id, resolve_manifest_path(root, entry.get("sam3_scene_dir", entry["sam3_dir"])))

    total_records = 0
    quality_ok = 0
    refined = 0
    box_like = 0
    matched_track = 0
    view_counts: dict[str, int] = {}
    for scene_id, scene_dir in sorted(scene_dirs.items()):
        for view_dir in sorted(scene_dir.glob("view_*")):
            path = view_dir / "instances.jsonl"
            if not path.exists():
                continue
            local_count = 0
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    total_records += 1
                    local_count += 1
                    quality_ok += int(bool(record.get("quality_ok", False)))
                    refined += int(bool(record.get("refined", False)))
                    box_like += int(bool(record.get("box_like_flag", False)))
                    matched_track += int(record.get("matched_track_id", -1) >= 0)
            view_counts[f"{scene_id}/{view_dir.name}"] = local_count
    return {
        "total_records": total_records,
        "quality_ok": quality_ok,
        "refined": refined,
        "box_like_flag": box_like,
        "matched_track": matched_track,
        "quality_ok_ratio": float(quality_ok / max(total_records, 1)),
        "box_like_ratio": float(box_like / max(total_records, 1)),
        "matched_track_ratio": float(matched_track / max(total_records, 1)),
        "views": view_counts,
    }


def audit_clip_sample(sample: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    cache = sample["dvgt_cache"]
    sup = sample["supervision"]
    full = sample["sam3_fullres"]

    visible = sup["track_visible"] > 0
    track_boxes = sup["track_boxes_3d"]
    visible_boxes = track_boxes[visible]
    invalid_box_size = int(np.sum((visible_boxes[:, 3:6] <= 0.0))) if visible_boxes.size else 0

    occ_label = sup["occ_label"]
    occ_dyn = sup["occ_dyn_label"]
    dyn_14 = sup["dyn_soft_mask_1_4"]
    full_dyn = full["sam3_dyn_mask_full"]
    full_dyn_14 = np.stack(
        [
            np.stack(
                [resize_mask_float(full_dyn[t, v], dyn_14.shape[-2:]).astype(np.float32) for v in range(full_dyn.shape[1])],
                axis=0,
            )
            for t in range(full_dyn.shape[0])
        ],
        axis=0,
    )
    bin_pred = dyn_14 > 0.5
    bin_full = full_dyn_14 > 0.5
    inter = float(np.logical_and(bin_pred, bin_full).sum())
    union = float(np.logical_or(bin_pred, bin_full).sum())

    return {
        "sample_index": sample_idx,
        "scene_id": sample["scene_id"],
        "clip_id": sample["clip_id"],
        "cache_shapes": {key: to_shape(value) for key, value in cache.items()},
        "supervision_shapes": {key: to_shape(value) for key, value in sup.items()},
        "fullres_shapes": {key: to_shape(value) for key, value in full.items()},
        "points_finite": bool(np.isfinite(cache["points"]).all()),
        "points_conf_range": [float(cache["points_conf"].min()), float(cache["points_conf"].max())],
        "track_visible_count": int(visible.sum()),
        "track_invalid_size_count": invalid_box_size,
        "occ_unique": sorted(int(x) for x in np.unique(occ_label)),
        "occ_dyn_unique": sorted(int(x) for x in np.unique(occ_dyn)),
        "occ_dyn_outside_occ": int(np.logical_and(occ_dyn > 0, occ_label <= 0).sum()),
        "dyn_soft_mask_range": [float(dyn_14.min()), float(dyn_14.max())],
        "inst_quality_mask_range": [float(sup["inst_quality_mask_1_8"].min()), float(sup["inst_quality_mask_1_8"].max())],
        "gs_global_track_pixels": int((sup["gs_global_track_id_1_8"] >= 0).sum()),
        "gs_local_inst_pixels": int((sup["gs_local_inst_id_1_8"] >= 0).sum()),
        "full_dyn_valid_pixels": int(np.logical_and(full["sam3_dyn_mask_full"] > 0.5, full["sam3_valid_mask_full"] > 0.5).sum()),
        "full_all_valid_pixels": int(np.logical_and(full["sam3_all_mask_full"] > 0.5, full["sam3_valid_mask_full"] > 0.5).sum()),
        "dyn14_vs_full14_iou": float(inter / union) if union > 0 else 0.0,
        "dyn14_vs_full14_mae": float(np.mean(np.abs(dyn_14 - full_dyn_14))),
    }


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    entries = load_manifest(args.manifest)
    dataset = DVGTOccClipDataset(
        manifest_path=args.manifest,
        root=root,
        load_cache=True,
        load_supervision=True,
        load_scene_sam3_full=False,
        projected_semantic_classes=DEFAULT_DVGT_OCC_CONFIG.projected_semantic_classes,
    )

    summary: dict[str, Any] = {
        "manifest_path": str(args.manifest),
        "output_root": str(root),
        "num_entries": len(entries),
        "base_checks": {
            "cache_metric_meta_ok": 0,
            "cache_metric_meta_bad": [],
            "points_all_finite": True,
            "occ_dyn_outside_occ_total": 0,
            "track_invalid_size_total": 0,
            "global_track_pixels_total": 0,
            "local_inst_pixels_total": 0,
        },
        "scene_level_sam3": audit_scene_level_records(entries, root),
        "sample_clips": [],
    }

    for entry in entries:
        cache_dir = resolve_manifest_path(root, entry["dvgt_cache_dir"])
        meta_path = cache_dir / "cache_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if bool(meta.get("cache_points_are_metric", False)) and float(meta.get("points_metric_scale", 1.0)) == 1.0:
                summary["base_checks"]["cache_metric_meta_ok"] += 1
            else:
                summary["base_checks"]["cache_metric_meta_bad"].append(str(cache_dir))

    for idx in range(len(dataset)):
        sample = dataset[idx]
        cache = sample["dvgt_cache"]
        sup = sample["supervision"]
        summary["base_checks"]["points_all_finite"] = bool(
            summary["base_checks"]["points_all_finite"] and np.isfinite(cache["points"]).all()
        )
        visible = sup["track_visible"] > 0
        visible_boxes = sup["track_boxes_3d"][visible]
        if visible_boxes.size:
            summary["base_checks"]["track_invalid_size_total"] += int(np.sum(visible_boxes[:, 3:6] <= 0.0))
        summary["base_checks"]["occ_dyn_outside_occ_total"] += int(
            np.logical_and(sup["occ_dyn_label"] > 0, sup["occ_label"] <= 0).sum()
        )
        summary["base_checks"]["global_track_pixels_total"] += int((sup["gs_global_track_id_1_8"] >= 0).sum())
        summary["base_checks"]["local_inst_pixels_total"] += int((sup["gs_local_inst_id_1_8"] >= 0).sum())

    sample_indices = evenly_spaced_indices(len(entries), args.sample_clips)
    sample_dataset = DVGTOccClipDataset(
        manifest_path=args.manifest,
        root=root,
        load_cache=True,
        load_supervision=True,
        load_scene_sam3_full=True,
        projected_semantic_classes=DEFAULT_DVGT_OCC_CONFIG.projected_semantic_classes,
    )
    for idx in sample_indices:
        summary["sample_clips"].append(audit_clip_sample(sample_dataset[idx], idx))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "num_entries": len(entries), "sample_indices": sample_indices}, ensure_ascii=False))


if __name__ == "__main__":
    main()
