#!/usr/bin/env python3
"""Extract processed sky masks with the SAM3 video predictor.

This intentionally keeps the contract simple:
- prompt only ``sky``
- no multi-class merging
- write processed masks to ``scene_output_dir/sky_masks_sam3``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.data.manifest import load_manifest, resolve_manifest_path
from dvgt_occ.data.stage_a_utils import collect_scene_image_paths, shard_items


SKY_PROMPT = "sky"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--sam3-ckpt", type=Path, required=True)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--view-ids", nargs="*", type=int, default=None)
    parser.add_argument("--window-len", type=int, default=96)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _preflight(sam3_root: Path, sam3_ckpt: Path) -> None:
    if not sam3_root.exists():
        raise FileNotFoundError(f"SAM3 code directory does not exist: {sam3_root}")
    if not sam3_ckpt.exists():
        raise FileNotFoundError(f"SAM3 checkpoint does not exist: {sam3_ckpt}")
    sys.path.insert(0, str(sam3_root.resolve()))


def _predictor_gpu_ids_from_env() -> list[int] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return None
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if not tokens:
        return None
    return list(range(len(tokens)))


def _write_view_symlinks(image_paths: list[Path], temp_dir: Path) -> None:
    for idx, image_path in enumerate(image_paths):
        os.symlink(image_path.resolve(), temp_dir / f"{idx:05d}.jpg")


def _window_starts(num_frames: int, window_len: int, overlap: int) -> list[int]:
    if window_len <= 0 or window_len >= num_frames:
        return [0]
    stride = max(window_len - overlap, 1)
    starts = list(range(0, max(num_frames - window_len, 0) + 1, stride))
    last = max(num_frames - window_len, 0)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _to_numpy(value) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _masks_from_output(output: dict) -> tuple[np.ndarray, np.ndarray]:
    masks = _to_numpy(output.get("out_binary_masks", np.zeros((0, 1, 1), dtype=np.uint8))).astype(bool)
    scores = _to_numpy(output.get("out_probs", np.zeros((masks.shape[0],), dtype=np.float32))).astype(np.float32)
    if masks.ndim == 2:
        masks = masks[None]
    return masks, scores


def _top_band_rows(height: int) -> int:
    return max(8, height // 12)


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=True)
    out = np.ones_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            component: list[tuple[int, int]] = []
            while stack:
                y, xx = stack.pop()
                component.append((y, xx))
                for ny, nx in ((y - 1, xx), (y + 1, xx), (y, xx - 1), (y, xx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            components.append(component)
    return components


def _keep_top_components(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    band = _top_band_rows(mask.shape[0])
    kept = np.zeros_like(mask, dtype=bool)
    for component in _connected_components(mask):
        min_y = min(y for y, _ in component)
        if min_y >= band:
            continue
        for y, x in component:
            kept[y, x] = True
    return kept


def _score_selected_union(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if masks.shape[0] == 0:
        return np.zeros((1, 1), dtype=bool)
    # QC-first mode: trust the predictor and expose the full candidate union.
    union = masks.any(axis=0)
    return union.astype(bool)


def _best_sky_mask(output: dict) -> np.ndarray:
    masks, scores = _masks_from_output(output)
    if masks.shape[0] == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    union = _score_selected_union(masks, scores)
    return union.astype(np.uint8) * 255


def _start_and_propagate(predictor, video_dir: Path) -> dict[int, dict]:
    response = predictor.handle_request(request={"type": "start_session", "resource_path": str(video_dir)})
    session_id = response["session_id"]
    predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": SKY_PROMPT,
        }
    )
    outputs: dict[int, dict] = {}
    for response in predictor.handle_stream_request(request={"type": "propagate_in_video", "session_id": session_id}):
        outputs[int(response["frame_index"])] = response.get("outputs", {})
    predictor.handle_request(request={"type": "close_session", "session_id": session_id})
    return outputs


def _iter_scene_view_jobs(clips: list[dict], output_root: Path, scene_ids: set[str] | None, view_ids: set[int] | None) -> list[tuple[str, Path, Path, int]]:
    seen: set[tuple[str, int]] = set()
    jobs: list[tuple[str, Path, Path, int]] = []
    for clip in clips:
        scene_id = str(clip["scene_id"])
        if scene_ids is not None and scene_id not in scene_ids:
            continue
        source_scene_dir = resolve_manifest_path(output_root, clip["source_scene_dir"])
        scene_output_dir = resolve_manifest_path(output_root, clip["scene_output_dir"])
        for view_id in clip["view_ids"]:
            view_id = int(view_id)
            if view_ids is not None and view_id not in view_ids:
                continue
            key = (scene_id, view_id)
            if key in seen:
                continue
            seen.add(key)
            jobs.append((scene_id, source_scene_dir, scene_output_dir, view_id))
    jobs.sort(key=lambda item: (item[0], item[3]))
    return jobs


def _save_meta(out_dir: Path, payload: dict, view_id: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"meta_view_{view_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    _preflight(args.sam3_root, args.sam3_ckpt)
    from sam3.model_builder import build_sam3_video_predictor

    clips = load_manifest(args.manifest)
    scene_ids = {str(scene_id).zfill(3) for scene_id in args.scene_ids} if args.scene_ids else None
    view_ids = set(int(view_id) for view_id in args.view_ids) if args.view_ids else None
    jobs = _iter_scene_view_jobs(clips, args.output_root, scene_ids, view_ids)
    jobs = shard_items(jobs, shard_index=args.shard_index, num_shards=args.num_shards)

    predictor = build_sam3_video_predictor(
        checkpoint_path=str(args.sam3_ckpt),
        gpus_to_use=_predictor_gpu_ids_from_env(),
    )
    try:
        for scene_id, source_scene_dir, scene_output_dir, view_id in jobs:
            out_dir = scene_output_dir / "sky_masks_sam3"
            out_dir.mkdir(parents=True, exist_ok=True)
            scene_frame_ids, scene_image_paths = collect_scene_image_paths(source_scene_dir, [view_id])
            image_paths = scene_image_paths[int(view_id)]
            starts = _window_starts(len(scene_frame_ids), args.window_len, args.overlap)
            num_written = 0
            for win_idx, start in enumerate(starts):
                end = len(scene_frame_ids) if args.window_len <= 0 else min(start + args.window_len, len(scene_frame_ids))
                emit_start = start if win_idx == 0 else start + args.overlap
                with tempfile.TemporaryDirectory(prefix=f"sky_{scene_id}_{view_id}_") as tmp_name:
                    tmp_dir = Path(tmp_name)
                    _write_view_symlinks(image_paths[start:end], tmp_dir)
                    outputs = _start_and_propagate(predictor, tmp_dir)
                for frame_offset, output in outputs.items():
                    scene_frame_idx = start + int(frame_offset)
                    if scene_frame_idx < emit_start:
                        continue
                    frame_id = int(scene_frame_ids[scene_frame_idx])
                    out_path = out_dir / f"{frame_id:03d}_{view_id}.png"
                    if out_path.exists() and not args.overwrite:
                        continue
                    mask = _best_sky_mask(output)
                    if mask.shape != np.array(Image.open(image_paths[scene_frame_idx]).convert("L")).shape:
                        # predictor should already emit original resolution; keep a safe fallback
                        h, w = Image.open(image_paths[scene_frame_idx]).convert("L").size[1], Image.open(image_paths[scene_frame_idx]).convert("L").size[0]
                        mask = np.asarray(Image.fromarray(mask, mode="L").resize((w, h), Image.Resampling.NEAREST))
                    Image.fromarray(mask, mode="L").save(out_path)
                    num_written += 1
            _save_meta(
                out_dir,
                {
                    "generator": "extract_sky_masks_sam3.py",
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "prompt": SKY_PROMPT,
                    "window_len": args.window_len,
                    "overlap": args.overlap,
                    "num_written_this_run": num_written,
                    "source_scene_dir": str(source_scene_dir),
                },
                view_id=view_id,
            )
            print(f"[sky_masks] scene={scene_id} view={view_id} wrote={num_written}")
    finally:
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()


if __name__ == "__main__":
    main(parse_args())
