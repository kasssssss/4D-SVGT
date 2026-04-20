#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit scene-level SAM3 assets view-by-view.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/nuscenes_dvgt_v0"),
        help="Stage-A output root containing trainval/scene_xxx/sam3_scene.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2026-04-19",
        help="Views with instances.jsonl older than this date are flagged as stale.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write audit JSON.",
    )
    return parser.parse_args()


def _iso_mtime(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _count_image_frames(scene_dir: Path, view_id: int) -> int:
    images_dir = scene_dir / "images"
    if not images_dir.exists():
        return 0
    return sum(1 for _ in images_dir.glob(f"*_{view_id}.jpg"))


def _scan_instances(instances_path: Path) -> dict[str, Any]:
    num_records = 0
    frames = set()
    local_ids = set()
    quality_ok = 0
    refined = 0
    box_like = 0
    class_counter: Counter[str] = Counter()
    exhaustive_false = 0

    with instances_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            num_records += 1
            frame_idx = int(row.get("scene_frame_index", -1))
            local_id = int(row.get("local_mask_id", -1))
            if frame_idx >= 0:
                frames.add(frame_idx)
            if local_id >= 0:
                local_ids.add(local_id)
            if bool(row.get("quality_ok", False)):
                quality_ok += 1
            if bool(row.get("refined", False)):
                refined += 1
            if bool(row.get("box_like_flag", False)):
                box_like += 1
            if not bool(row.get("exhaustive_ok", True)):
                exhaustive_false += 1
            class_counter[str(row.get("class_name", "unknown"))] += 1

    return {
        "num_records": num_records,
        "num_frames_with_records": len(frames),
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "num_local_ids": len(local_ids),
        "quality_ok_records": quality_ok,
        "refined_records": refined,
        "box_like_records": box_like,
        "exhaustive_false_records": exhaustive_false,
        "class_counter": dict(sorted(class_counter.items())),
    }


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.resolve()
    cutoff_ts = datetime.fromisoformat(args.cutoff_date).timestamp()

    trainval_root = output_root / "trainval"
    scene_dirs = sorted(p for p in trainval_root.glob("scene_*") if p.is_dir())

    views: list[dict[str, Any]] = []
    stale_views: list[str] = []
    missing_progress: list[str] = []
    not_done: list[str] = []
    missing_instances: list[str] = []
    suspicious: list[str] = []

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        scene_sam3 = scene_dir / "sam3_scene"
        for view_id in range(6):
            tag = f"{scene_name}/view_{view_id}"
            view_dir = scene_sam3 / f"view_{view_id}"
            progress_path = view_dir / "progress.json"
            instances_path = view_dir / "instances.jsonl"
            partial_path = view_dir / "instances.partial.jsonl"

            progress: dict[str, Any] | None = None
            if progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))

            image_frames = _count_image_frames(scene_dir, view_id)
            row: dict[str, Any] = {
                "tag": tag,
                "scene": scene_name,
                "view_id": view_id,
                "image_frames": image_frames,
                "progress_exists": progress_path.exists(),
                "progress_status": None if progress is None else progress.get("status"),
                "completed_windows": None if progress is None else progress.get("completed_windows"),
                "num_windows": None if progress is None else progress.get("num_windows"),
                "progress_mtime": _iso_mtime(progress_path),
                "instances_exists": instances_path.exists(),
                "instances_mtime": _iso_mtime(instances_path),
                "partial_exists": partial_path.exists(),
                "partial_mtime": _iso_mtime(partial_path),
            }

            if not progress_path.exists():
                missing_progress.append(tag)
            elif progress.get("status") != "done":
                not_done.append(tag)

            if not instances_path.exists():
                missing_instances.append(tag)
            else:
                inst_stats = _scan_instances(instances_path)
                row.update(inst_stats)
                if instances_path.stat().st_mtime < cutoff_ts:
                    stale_views.append(tag)
                if inst_stats["num_records"] == 0:
                    suspicious.append(f"{tag}:empty_instances")
                if image_frames and inst_stats["frame_max"] is not None and inst_stats["frame_max"] >= image_frames:
                    suspicious.append(f"{tag}:frame_index_out_of_range")
            views.append(row)

    summary = {
        "output_root": str(output_root),
        "cutoff_date": args.cutoff_date,
        "num_scenes": len(scene_dirs),
        "num_views": len(views),
        "done_views": sum(1 for v in views if v["progress_status"] == "done"),
        "progress_missing_views": len(missing_progress),
        "not_done_views": len(not_done),
        "instances_missing_views": len(missing_instances),
        "stale_views": stale_views,
        "missing_progress": missing_progress,
        "not_done": not_done,
        "missing_instances": missing_instances,
        "suspicious": suspicious,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "views": views}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
