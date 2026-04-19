#!/usr/bin/env python3
"""Visualize SAM3 local-mask continuity across a window boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def rle_to_binary_mask(rle: dict) -> np.ndarray:
    height, width = (int(v) for v in rle["size"])
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    cursor = 0
    value = 0
    for count in rle["counts"]:
        count = int(count)
        if value == 1 and count > 0:
            flat[cursor : cursor + count] = 1
        cursor += count
        value = 1 - value
    return flat.reshape((height, width), order="F").astype(bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=str, required=True)
    parser.add_argument("--view-id", type=int, required=True)
    parser.add_argument("--local-mask-id", type=int, required=True)
    parser.add_argument("--frame-start", type=int, default=60)
    parser.add_argument("--frame-end", type=int, default=80)
    parser.add_argument(
        "--selected-frames",
        type=int,
        nargs="*",
        default=[60, 62, 63, 64, 65, 68, 72, 76, 80],
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-scene-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_id = f"{int(args.scene_id):03d}" if args.scene_id.isdigit() else args.scene_id
    view_dir = args.output_root / "trainval" / f"scene_{scene_id}" / "sam3_scene" / f"view_{args.view_id}"
    scene_dir = args.source_scene_dir
    inst_path = view_dir / "instances.jsonl"
    if not inst_path.exists():
        raise FileNotFoundError(f"Missing instances jsonl: {inst_path}")

    records = []
    with inst_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if (
                int(rec["local_mask_id"]) == args.local_mask_id
                and args.frame_start <= int(rec["scene_frame_index"]) <= args.frame_end
            ):
                records.append(rec)
    records.sort(key=lambda item: int(item["scene_frame_index"]))
    if not records:
        raise RuntimeError("No records found for requested local-mask id/frame range")

    frame_to_rec = {int(rec["scene_frame_index"]): rec for rec in records}
    selected = [frame for frame in args.selected_frames if frame in frame_to_rec]
    if not selected:
        raise RuntimeError("No selected frames overlap requested record range")

    thumbs = []
    for frame_idx in selected:
        rec = frame_to_rec[frame_idx]
        img_path = scene_dir / "images" / f"{frame_idx:03d}_{args.view_id}.jpg"
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = rle_to_binary_mask(rec["rle"])
        overlay = img.copy().astype(np.float32)
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([255, 40, 40], dtype=np.float32)
        pil = Image.fromarray(overlay.astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        x1, y1, x2, y2 = rec["bbox_mask_tight"]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 255), width=4)
        pil = pil.resize((480, 270))
        draw = ImageDraw.Draw(pil)
        draw.rectangle((0, 0, 479, 30), fill=(0, 0, 0))
        draw.text(
            (8, 8),
            f"frame {frame_idx} | id {args.local_mask_id} | area {rec['visible_area']}",
            fill=(255, 255, 255),
        )
        thumbs.append(pil)

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * 480, rows * 270 + 340), (255, 255, 255))
    for idx, thumb in enumerate(thumbs):
        canvas.paste(thumb, ((idx % cols) * 480, (idx // cols) * 270))

    cls_name = records[0]["class_name"]
    draw = ImageDraw.Draw(canvas)
    lines = [
        f"scene_{scene_id} view_{args.view_id} local_mask_id={args.local_mask_id} class={cls_name}",
        f"present frames: {min(frame_to_rec)}..{max(frame_to_rec)} count={len(records)}",
        f"crosses 64 boundary: {min(frame_to_rec) <= 63 and max(frame_to_rec) >= 64}",
        f"all frames present in {args.frame_start}..{args.frame_end}: "
        f"{len(records) == (args.frame_end - args.frame_start + 1)}",
    ]
    for frame_idx in range(args.frame_start, args.frame_end + 1):
        if frame_idx in frame_to_rec:
            rec = frame_to_rec[frame_idx]
            lines.append(
                f"frame {frame_idx}: present area={rec['visible_area']} "
                f"q={rec['quality_ok']} refined={rec['refined']}"
            )
        else:
            lines.append(f"frame {frame_idx}: absent")
    y = rows * 270 + 10
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0))
        y += 14

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"scene_{scene_id}_view_{args.view_id}_local_{args.local_mask_id}"
        f"_frames_{args.frame_start}_{args.frame_end}"
    )
    out_png = args.out_dir / f"{stem}.png"
    out_json = args.out_dir / f"{stem}.json"
    canvas.save(out_png)
    out_json.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "view_id": args.view_id,
                "local_mask_id": args.local_mask_id,
                "class_name": cls_name,
                "present_frames": sorted(frame_to_rec),
                "num_present_frames": len(records),
                "crosses_window_boundary_64": bool(min(frame_to_rec) <= 63 and max(frame_to_rec) >= 64),
                "all_frames_present": bool(len(records) == (args.frame_end - args.frame_start + 1)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_png)
    print(out_json)


if __name__ == "__main__":
    main()
