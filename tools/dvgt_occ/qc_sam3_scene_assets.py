#!/usr/bin/env python3
"""Direct scene-level SAM3 QC that does not depend on clip-index rebuild."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMAGE_RE = re.compile(r"^(?P<frame>\d+)_(?P<view>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def resolve_manifest_path(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(root) / path).resolve()


def collect_scene_image_paths(scene_dir: Path, view_ids: list[int]) -> tuple[list[int], dict[int, list[Path]]]:
    by_frame: dict[int, dict[int, Path]] = defaultdict(dict)
    for path in sorted((scene_dir / "images").iterdir()):
        if not path.is_file():
            continue
        match = IMAGE_RE.match(path.name)
        if match is None:
            continue
        frame_id = int(match.group("frame"))
        view_id = int(match.group("view"))
        if view_id in view_ids:
            by_frame[frame_id][view_id] = path
    frames = [frame_id for frame_id, frame_views in sorted(by_frame.items()) if set(view_ids).issubset(frame_views)]
    paths = {view_id: [by_frame[frame_id][view_id] for frame_id in frames] for view_id in view_ids}
    return frames, paths


def rle_to_binary_mask(rle: dict[str, Any]) -> np.ndarray:
    size = tuple(int(v) for v in rle["size"])
    flat = np.zeros(size[0] * size[1], dtype=np.uint8)
    cursor = 0
    value = 0
    for run in rle["counts"]:
        run = int(run)
        if run > 0 and value == 1:
            flat[cursor : cursor + run] = 1
        cursor += run
        value = 1 - value
    return flat.reshape(size, order="F").astype(bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and visualize scene-level SAM3 assets per view.")
    parser.add_argument("--manifest", type=Path, default=Path("data/nuscenes_dvgt_v0/manifest_trainval.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/nuscenes_dvgt_v0"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/nuscenes_dvgt_v0/qc/sam3_scene_assets_20260419"),
    )
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=426)
    parser.add_argument("--scene-ids", type=str, nargs="*", default=None)
    parser.add_argument("--view-ids", type=int, nargs="*", default=None)
    return parser.parse_args()


def _load_manifest_clips(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "clips" in payload:
        return list(payload["clips"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"Unsupported manifest format at {path}")


def _scene_specs(
    clips: list[dict[str, Any]],
    output_root: Path,
    allowed_scene_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    by_scene: dict[str, dict[str, Any]] = {}
    for clip in clips:
        scene_id = str(clip["scene_id"])
        if allowed_scene_ids is not None and scene_id not in allowed_scene_ids:
            continue
        if scene_id in by_scene:
            continue
        source_scene_dir = resolve_manifest_path(output_root, str(clip["source_scene_dir"]))
        by_scene[scene_id] = {
            "scene_id": scene_id,
            "scene_output_dir": resolve_manifest_path(output_root, str(clip["scene_output_dir"])),
            "source_scene_dir": source_scene_dir,
            "scene_frame_ids_10hz": [int(frame_id) for frame_id in clip["scene_frame_ids_10hz"]],
        }
    return [by_scene[key] for key in sorted(by_scene)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "completed_windows": 0, "num_windows": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _resize_mask(mask: np.ndarray, size_hw: tuple[int, int], resample: int = Image.Resampling.NEAREST) -> np.ndarray:
    mask_img = Image.fromarray(np.clip(mask.astype(np.float32) * 255.0, 0, 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize((size_hw[1], size_hw[0]), resample)
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def _render_rgb(image_path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((size_hw[1], size_hw[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    mask3 = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
    return np.clip(base_rgb * (1.0 - alpha * mask3) + color_arr * (alpha * mask3), 0.0, 1.0)


def _rgb_to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def _records_by_frame(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_frame[int(row["frame_id_10hz"])].append(row)
    return by_frame


def _frame_union_masks(
    frame_records: list[dict[str, Any]],
    size_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    accepted = np.zeros(size_hw, dtype=np.float32)
    rejected = np.zeros(size_hw, dtype=np.float32)
    refined = np.zeros(size_hw, dtype=np.float32)
    stats = {"accepted": 0, "rejected": 0, "refined": 0, "box_like": 0}
    for row in frame_records:
        mask = rle_to_binary_mask(row["rle"]).astype(np.float32)
        if mask.shape != size_hw:
            mask = _resize_mask(mask, size_hw)
        if bool(row.get("quality_ok", False)) and not bool(row.get("box_like_flag", False)):
            accepted = np.maximum(accepted, mask)
            stats["accepted"] += 1
        else:
            rejected = np.maximum(rejected, mask)
            stats["rejected"] += 1
        if bool(row.get("refined", False)):
            refined = np.maximum(refined, mask)
            stats["refined"] += 1
        if bool(row.get("box_like_flag", False)):
            stats["box_like"] += 1
    return accepted, rejected, refined, stats


def _representative_frame(
    frame_to_records: dict[int, list[dict[str, Any]]],
    size_hw: tuple[int, int],
) -> tuple[int | None, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    best_frame_id: int | None = None
    best_score = -1.0
    for frame_id, rows in frame_to_records.items():
        accepted_area = sum(
            float(row.get("visible_area", 0))
            for row in rows
            if bool(row.get("quality_ok", False)) and not bool(row.get("box_like_flag", False))
        )
        rejected_area = sum(
            float(row.get("visible_area", 0))
            for row in rows
            if not (bool(row.get("quality_ok", False)) and not bool(row.get("box_like_flag", False)))
        )
        score = accepted_area + 0.25 * rejected_area
        if score > best_score:
            best_score = score
            best_frame_id = int(frame_id)
    if best_frame_id is None:
        empty = np.zeros(size_hw, dtype=np.float32)
        return None, empty, empty, empty, {"accepted": 0, "rejected": 0, "refined": 0, "box_like": 0}
    accepted, rejected, refined, stats = _frame_union_masks(frame_to_records[best_frame_id], size_hw)
    return best_frame_id, accepted, rejected, refined, stats


def _view_summary(scene_id: str, view_id: int, records: list[dict[str, Any]], progress: dict[str, Any]) -> dict[str, Any]:
    class_hist = Counter(str(row.get("class_name", "unknown")) for row in records)
    quality_ok = sum(int(bool(row.get("quality_ok", False))) for row in records)
    box_like = sum(int(bool(row.get("box_like_flag", False))) for row in records)
    refined = sum(int(bool(row.get("refined", False))) for row in records)
    frames = sorted({int(row["frame_id_10hz"]) for row in records})
    local_ids = sorted({int(row["local_mask_id"]) for row in records})
    return {
        "scene_id": scene_id,
        "view_id": int(view_id),
        "progress_status": str(progress.get("status", "missing")),
        "completed_windows": int(progress.get("completed_windows", 0)),
        "num_windows": int(progress.get("num_windows", 0)),
        "num_records": int(len(records)),
        "quality_ok_records": int(quality_ok),
        "box_like_records": int(box_like),
        "refined_records": int(refined),
        "num_frames_covered": int(len(frames)),
        "min_frame_id_10hz": int(frames[0]) if frames else None,
        "max_frame_id_10hz": int(frames[-1]) if frames else None,
        "num_local_mask_ids": int(len(local_ids)),
        "class_hist": dict(sorted(class_hist.items())),
    }


def _save_view_overlay(
    scene_id: str,
    view_id: int,
    image_path: Path | None,
    frame_id: int | None,
    accepted: np.ndarray,
    rejected: np.ndarray,
    refined: np.ndarray,
    stats: dict[str, int],
    summary: dict[str, Any],
    out_path: Path,
    size_hw: tuple[int, int],
) -> None:
    if image_path is None or frame_id is None:
        rgb = np.zeros((size_hw[0], size_hw[1], 3), dtype=np.float32)
        title = f"scene={scene_id} view={view_id} | no representative frame"
    else:
        rgb = _render_rgb(image_path, size_hw)
        title = f"scene={scene_id} view={view_id} frame={frame_id}"

    accepted_panel = _overlay_mask(rgb, accepted, (0.0, 1.0, 0.0), 0.45)
    rejected_panel = _overlay_mask(rgb, rejected, (1.0, 0.0, 0.0), 0.55)
    rejected_panel = _overlay_mask(rejected_panel, refined, (0.1, 0.4, 1.0), 0.35)
    composite_panel = _overlay_mask(rgb, accepted, (0.0, 1.0, 0.0), 0.40)
    composite_panel = _overlay_mask(composite_panel, rejected, (1.0, 0.0, 0.0), 0.40)
    composite_panel = _overlay_mask(composite_panel, refined, (0.1, 0.4, 1.0), 0.25)

    panels = [
        ("accepted mask", _rgb_to_image(accepted_panel)),
        ("rejected(red) + refined(blue)", _rgb_to_image(rejected_panel)),
        (
            f"records={summary['num_records']} ok={summary['quality_ok_records']} "
            f"box={summary['box_like_records']} ref={summary['refined_records']}",
            _rgb_to_image(composite_panel),
        ),
    ]

    header_h = 48
    label_h = 22
    canvas = Image.new("RGB", (size_hw[1] * 3, header_h + label_h + size_hw[0]), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 8),
        (
            f"{title} | progress={summary['progress_status']} {summary['completed_windows']}/{summary['num_windows']} | "
            f"rep accepted={stats['accepted']} rejected={stats['rejected']} box={stats['box_like']}"
        ),
        fill=(240, 240, 240),
    )
    for idx, (label, panel) in enumerate(panels):
        x0 = idx * size_hw[1]
        draw.text((x0 + 8, header_h + 2), label, fill=(220, 220, 220))
        canvas.paste(panel, (x0, header_h + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _save_overview_grid(
    summaries: list[dict[str, Any]],
    overlay_paths: dict[tuple[str, int], Path],
    out_path: Path,
) -> None:
    scenes = sorted({str(row["scene_id"]) for row in summaries})
    view_ids = sorted({int(row["view_id"]) for row in summaries})
    cell_w = 320
    cell_h = 140
    grid = Image.new("RGB", (len(view_ids) * cell_w, len(scenes) * cell_h), color=(12, 12, 12))
    draw = ImageDraw.Draw(grid)
    for row_idx, scene_id in enumerate(scenes):
        for col_idx, view_id in enumerate(view_ids):
            overlay_path = overlay_paths.get((scene_id, view_id))
            summary = next(item for item in summaries if str(item["scene_id"]) == scene_id and int(item["view_id"]) == view_id)
            x0 = col_idx * cell_w
            y0 = row_idx * cell_h
            if overlay_path is not None and overlay_path.exists():
                thumb = Image.open(overlay_path).convert("RGB")
                thumb = thumb.resize((cell_w, cell_h - 34), Image.Resampling.BILINEAR)
                grid.paste(thumb, (x0, y0 + 34))
            else:
                grid.paste(Image.new("RGB", (cell_w, cell_h - 34), color=(30, 30, 30)), (x0, y0 + 34))
            draw.text(
                (x0 + 6, y0 + 6),
                f"s{scene_id}v{view_id}\n{summary['progress_status']} "
                f"ok={summary['quality_ok_records']} box={summary['box_like_records']}",
                fill=(235, 235, 235),
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> None:
    args = parse_args()
    manifest = _load_manifest_clips(args.manifest)
    output_root = args.output_root.resolve()
    out_dir = args.out_dir.resolve()
    overlays_dir = out_dir / "by_view"
    size_hw = (args.image_height, args.image_width)
    allowed_scene_ids = None if not args.scene_ids else {str(scene_id) for scene_id in args.scene_ids}
    allowed_view_ids = list(range(6)) if not args.view_ids else sorted({int(view_id) for view_id in args.view_ids})

    summaries: list[dict[str, Any]] = []
    overlay_paths: dict[tuple[str, int], Path] = {}

    for spec in _scene_specs(manifest, output_root, allowed_scene_ids=allowed_scene_ids):
        frame_ids, image_paths = collect_scene_image_paths(spec["source_scene_dir"], view_ids=allowed_view_ids)
        image_by_view_frame = {
            view_id: {int(frame_id): path for frame_id, path in zip(frame_ids, paths)}
            for view_id, paths in image_paths.items()
        }
        sam3_scene_dir = spec["scene_output_dir"] / "sam3_scene"
        for view_id in allowed_view_ids:
            view_dir = sam3_scene_dir / f"view_{view_id}"
            records = _load_jsonl(view_dir / "instances.jsonl")
            progress = _load_progress(view_dir / "progress.json")
            summary = _view_summary(spec["scene_id"], view_id, records, progress)
            frame_to_records = _records_by_frame(records)
            rep_frame, accepted, rejected, refined, rep_stats = _representative_frame(frame_to_records, size_hw)
            summary["representative_frame_id_10hz"] = rep_frame
            summary["representative_accepted_area"] = int(accepted.sum())
            summary["representative_rejected_area"] = int(rejected.sum())
            summary["representative_refined_area"] = int(refined.sum())
            overlay_path = overlays_dir / f"scene_{spec['scene_id']}_view_{view_id}.png"
            _save_view_overlay(
                spec["scene_id"],
                view_id,
                image_by_view_frame.get(view_id, {}).get(rep_frame) if rep_frame is not None else None,
                rep_frame,
                accepted,
                rejected,
                refined,
                rep_stats,
                summary,
                overlay_path,
                size_hw,
            )
            summaries.append(summary)
            overlay_paths[(spec["scene_id"], view_id)] = overlay_path

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    overview_path = out_dir / "overview_grid.png"
    _save_overview_grid(summaries, overlay_paths, overview_path)

    agg = {
        "num_views": len(summaries),
        "num_done": sum(1 for row in summaries if row["progress_status"] == "done"),
        "num_in_progress": sum(1 for row in summaries if row["progress_status"] == "in_progress"),
        "num_missing": sum(1 for row in summaries if row["progress_status"] == "missing"),
        "num_zero_record_views": sum(1 for row in summaries if row["num_records"] == 0),
        "num_box_like_nonzero_views": sum(1 for row in summaries if row["box_like_records"] > 0),
        "overview_grid": str(overview_path),
        "summary_json": str(summary_path),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
