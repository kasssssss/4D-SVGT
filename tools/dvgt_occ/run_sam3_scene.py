#!/usr/bin/env python3
"""Run scene-level SAM3 video segmentation and store primary mask assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.data.manifest import load_manifest
from dvgt_occ.data.stage_a_utils import (
    binary_mask_to_rle,
    bbox_iou_xyxy,
    build_track_arrays,
    centroid_in_bbox,
    collect_scene_image_paths,
    distributed_shard_spec,
    entry_output_dirs,
    filter_manifest_entries,
    manifest_scene_groups,
    mask_tight_bbox_xyxy,
    rle_to_binary_mask,
    shard_items,
    xyxy_to_xywh,
)


DEFAULT_PROMPTS = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction vehicle",
    "pedestrian",
    "bicycle",
    "motorcycle",
    "cyclist",
    "bicyclist",
    "motorcyclist",
)

PROMPT_TO_CLASS = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "trailer": 3,
    "construction vehicle": 4,
    "pedestrian": 5,
    "bicycle": 6,
    "motorcycle": 7,
    "cyclist": 8,
    "bicyclist": 8,
    "motorcyclist": 8,
}


def _preflight(sam3_root: Path, sam3_ckpt: Path) -> None:
    if not sam3_root.exists():
        raise FileNotFoundError(f"SAM3 code directory does not exist: {sam3_root}")
    if not sam3_ckpt.exists():
        raise FileNotFoundError(f"SAM3 checkpoint does not exist: {sam3_ckpt}")
    sys.path.insert(0, str(sam3_root.resolve()))


def _stabilize_thread_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _install_sam3_import_shims() -> None:
    _stabilize_thread_env()
    if "huggingface_hub" not in sys.modules:
        hf_mod = types.ModuleType("huggingface_hub")

        def _hf_hub_download(*args, **kwargs):
            raise RuntimeError("huggingface_hub is not installed; pass a local --sam3-ckpt path instead")

        hf_mod.hf_hub_download = _hf_hub_download
        sys.modules.setdefault("huggingface_hub", hf_mod)
    if "iopath.common.file_io" not in sys.modules:
        iopath_mod = types.ModuleType("iopath")
        common_mod = types.ModuleType("iopath.common")
        file_io_mod = types.ModuleType("iopath.common.file_io")

        class _PathManager:
            @staticmethod
            def open(path, mode="r", *args, **kwargs):
                return open(path, mode, *args, **kwargs)

            @staticmethod
            def exists(path):
                return Path(path).exists()

            @staticmethod
            def isfile(path):
                return Path(path).is_file()

        file_io_mod.g_pathmgr = _PathManager()
        sys.modules.setdefault("iopath", iopath_mod)
        sys.modules.setdefault("iopath.common", common_mod)
        sys.modules.setdefault("iopath.common.file_io", file_io_mod)
    if "timm" not in sys.modules:
        import torch
        import torch.nn as nn

        timm_mod = types.ModuleType("timm")
        timm_layers_mod = types.ModuleType("timm.layers")
        timm_models_mod = types.ModuleType("timm.models")
        timm_models_layers_mod = types.ModuleType("timm.models.layers")

        class _DropPath(nn.Module):
            def __init__(self, drop_prob=0.0):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, x):
                return x

        def _trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
            with torch.no_grad():
                tensor.normal_(mean=mean, std=std)
                tensor.clamp_(min=a, max=b)
            return tensor

        for mod in (timm_layers_mod, timm_models_layers_mod):
            mod.DropPath = _DropPath
            mod.trunc_normal_ = _trunc_normal_
        timm_mod.layers = timm_layers_mod
        timm_models_mod.layers = timm_models_layers_mod
        sys.modules.setdefault("timm", timm_mod)
        sys.modules.setdefault("timm.layers", timm_layers_mod)
        sys.modules.setdefault("timm.models", timm_models_mod)
        sys.modules.setdefault("timm.models.layers", timm_models_layers_mod)
    if "ftfy" not in sys.modules:
        ftfy_mod = types.ModuleType("ftfy")
        ftfy_mod.fix_text = lambda text, *args, **kwargs: text
        sys.modules.setdefault("ftfy", ftfy_mod)


def _configure_runtime_cache(
    output_root: Path,
    shard_index: int,
    runtime_cache_root: Path | None = None,
) -> Path:
    cache_root = (runtime_cache_root if runtime_cache_root is not None else output_root / "runtime_cache" / "sam3_scene").resolve()
    shard_root = cache_root / f"shard_{shard_index:02d}"
    triton_dir = shard_root / "triton"
    xdg_dir = shard_root / "xdg"
    inductor_dir = shard_root / "torchinductor"
    temp_dir = shard_root / "tmp"
    for path in (triton_dir, xdg_dir, inductor_dir, temp_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
    os.environ["XDG_CACHE_HOME"] = str(xdg_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)
    return temp_dir


def _predictor_gpu_ids_from_env() -> list[int] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return None
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if not tokens:
        return None
    return list(range(len(tokens)))


def _to_numpy(value) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _propagate_session(predictor, session_id: str) -> Dict[int, dict]:
    outputs = {}
    for response in predictor.handle_stream_request(request={"type": "propagate_in_video", "session_id": session_id}):
        outputs[int(response["frame_index"])] = response.get("outputs", {})
    return outputs


def _write_view_symlinks(image_paths: List[Path], temp_dir: Path) -> None:
    for idx, image_path in enumerate(image_paths):
        os.symlink(image_path.resolve(), temp_dir / f"{idx:05d}.jpg")


def _masks_from_output(output: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks = _to_numpy(output.get("out_binary_masks", np.zeros((0, 1, 1), dtype=np.uint8))).astype(bool)
    boxes = _to_numpy(output.get("out_boxes_xywh", np.zeros((masks.shape[0], 4), dtype=np.float32))).astype(np.float32)
    scores = _to_numpy(output.get("out_probs", np.zeros((masks.shape[0],), dtype=np.float32))).astype(np.float32)
    obj_ids = _to_numpy(output.get("out_obj_ids", np.arange(masks.shape[0]))).astype(np.int64)
    if masks.ndim == 2:
        masks = masks[None]
    return masks, boxes, scores, obj_ids


def _window_starts(num_frames: int, window_len: int, overlap: int) -> list[int]:
    if window_len <= 0 or window_len >= num_frames:
        return [0]
    stride = max(window_len - overlap, 1)
    starts = list(range(0, max(num_frames - window_len, 0) + 1, stride))
    last = max(num_frames - window_len, 0)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _mask_box_fill_ratio(mask: np.ndarray) -> float:
    box = mask_tight_bbox_xyxy(mask)
    width = max(box[2] - box[0] + 1, 1)
    height = max(box[3] - box[1] + 1, 1)
    return float(mask.sum() / max(width * height, 1))


def _quality_flags(mask: np.ndarray, refined: bool = False) -> tuple[bool, bool, str]:
    if mask.sum() == 0:
        return False, False, "empty_mask"
    box_like_flag = _mask_box_fill_ratio(mask) > 0.92
    quality_ok = bool(mask.sum() >= 16 and not box_like_flag)
    if box_like_flag:
        return quality_ok, box_like_flag, "box_like_after_refine" if refined else "box_like"
    if mask.sum() < 16:
        return quality_ok, box_like_flag, "too_small"
    return quality_ok, box_like_flag, "ok"


def _clip_point_xy(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (
        float(np.clip(x, 0.0, max(width - 1, 0))),
        float(np.clip(y, 0.0, max(height - 1, 0))),
    )


def _mask_centroid_xy(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        height, width = mask.shape
        return float(width * 0.5), float(height * 0.5)
    return float(xs.mean()), float(ys.mean())


def _build_refine_points(mask: np.ndarray) -> tuple[list[list[float]], list[int]]:
    height, width = mask.shape
    x0, y0, x1, y1 = mask_tight_bbox_xyxy(mask)
    cx, cy = _mask_centroid_xy(mask)
    margin = max(int(round(max(x1 - x0 + 1, y1 - y0 + 1) * 0.06)), 3)
    neg_points = [
        _clip_point_xy(x0 - margin, cy, width, height),
        _clip_point_xy(x1 + margin, cy, width, height),
        _clip_point_xy(cx, y0 - margin, width, height),
        _clip_point_xy(cx, y1 + margin, width, height),
    ]
    pos_point = _clip_point_xy(cx, cy, width, height)
    denom = np.array([max(width, 1), max(height, 1)], dtype=np.float32)
    points = [list((np.array(pos_point, dtype=np.float32) / denom).tolist())]
    points.extend(list((np.array(point, dtype=np.float32) / denom).tolist()) for point in neg_points)
    labels = [1] + [0] * len(neg_points)
    return points, labels


def _collect_box_like_candidates(prompt_outputs: Mapping[int, dict]) -> dict[int, dict]:
    candidates: dict[int, dict] = {}
    for frame_offset, output in prompt_outputs.items():
        masks, boxes, scores, obj_ids = _masks_from_output(output)
        for obj_idx in range(masks.shape[0]):
            mask = masks[obj_idx]
            _, box_like_flag, _ = _quality_flags(mask, refined=False)
            if not box_like_flag:
                continue
            obj_id = int(obj_ids[obj_idx])
            area = int(mask.sum())
            candidate = candidates.get(obj_id)
            if candidate is None or area > candidate["area"]:
                candidates[obj_id] = {
                    "frame_offset": int(frame_offset),
                    "mask": mask,
                    "area": area,
                    "score": float(scores[obj_idx]) if obj_idx < len(scores) else 0.0,
                }
    return candidates


def _refine_box_like_tracks(predictor, session_id: str, prompt_outputs: Mapping[int, dict]) -> tuple[Dict[int, dict], set[int]]:
    candidates = _collect_box_like_candidates(prompt_outputs)
    if not candidates:
        return dict(prompt_outputs), set()
    refined_obj_ids: set[int] = set()
    for obj_id, candidate in sorted(candidates.items(), key=lambda item: (item[1]["frame_offset"], item[0])):
        points, point_labels = _build_refine_points(candidate["mask"])
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": int(candidate["frame_offset"]),
                "points": points,
                "point_labels": point_labels,
                "obj_id": int(obj_id),
            }
        )
        refined_obj_ids.add(int(obj_id))
    return _propagate_session(predictor, session_id), refined_obj_ids


def _start_and_propagate(predictor, video_dir: Path, prompt: str) -> tuple[Dict[int, dict], set[int]]:
    response = predictor.handle_request(request={"type": "start_session", "resource_path": str(video_dir)})
    session_id = response["session_id"]
    predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        }
    )
    outputs = _propagate_session(predictor, session_id)
    outputs, refined_obj_ids = _refine_box_like_tracks(predictor, session_id, outputs)
    predictor.handle_request(request={"type": "close_session", "session_id": session_id})
    return outputs, refined_obj_ids


def _track_box_compatibility(prompt: str, track_cls: int) -> float:
    if prompt in {"cyclist", "bicyclist", "motorcyclist"}:
        return 1.0 if int(track_cls) in {PROMPT_TO_CLASS["bicycle"], PROMPT_TO_CLASS["motorcycle"]} else 0.0
    expected = PROMPT_TO_CLASS.get(prompt, -1)
    return 1.0 if expected == int(track_cls) else 0.0


def _xywh_to_xyxy(box_xywh: np.ndarray) -> list[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + w, y + h]


def _box_corners(box: np.ndarray) -> np.ndarray:
    cx, cy, cz, dx, dy, dz, yaw = box.astype(np.float64)
    corners = np.array(
        [
            [sx * dx * 0.5, sy * dy * 0.5, sz * dz * 0.5]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ],
        dtype=np.float64,
    )
    rot = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return corners @ rot.T + np.array([cx, cy, cz], dtype=np.float64)


def _project_box_xyxy(scene_dir: Path, frame_id: int, view_id: int, first_pose: np.ndarray, box: np.ndarray) -> list[float] | None:
    intrinsic = np.loadtxt(scene_dir / "intrinsics" / f"{view_id}.txt", dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = intrinsic[:4]
    cam_to_world = np.loadtxt(scene_dir / "extrinsics" / f"{frame_id:03d}_{view_id}.txt", dtype=np.float64).reshape(4, 4)
    with Image.open(scene_dir / "images" / f"{frame_id:03d}_{view_id}.jpg") as image:
        width, height = image.size
    first_to_cam = np.linalg.inv(cam_to_world) @ first_pose
    corners = _box_corners(box)
    homo = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    cam = (first_to_cam @ homo.T).T[:, :3]
    valid = cam[:, 2] > 1e-3
    if valid.sum() < 2:
        return None
    uv = np.empty((valid.sum(), 2), dtype=np.float64)
    uv[:, 0] = cam[valid, 0] * fx / cam[valid, 2] + cx
    uv[:, 1] = cam[valid, 1] * fy / cam[valid, 2] + cy
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    x0, x1 = float(np.clip(x0, 0, width - 1)), float(np.clip(x1, 0, width - 1))
    y0, y1 = float(np.clip(y0, 0, height - 1)), float(np.clip(y1, 0, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _match_window_track(
    prompt: str,
    track_frames: Mapping[int, np.ndarray],
    stitched_tracks: Mapping[int, dict],
    overlap_start: int,
) -> int | None:
    best_id = None
    best_score = 0.0
    for local_id, stitched in stitched_tracks.items():
        if stitched["class_name"] != prompt:
            continue
        common_frames = [frame_idx for frame_idx in track_frames if frame_idx >= overlap_start and frame_idx in stitched["frames"]]
        if not common_frames:
            continue
        ious = []
        for frame_idx in common_frames:
            current = track_frames[frame_idx]
            previous = stitched["frames"][frame_idx]
            inter = np.logical_and(current, previous).sum()
            union = np.logical_or(current, previous).sum()
            ious.append(float(inter / max(union, 1)))
        score = float(np.mean(ious)) if ious else 0.0
        if score > best_score:
            best_score = score
            best_id = local_id
    if best_score >= 0.2:
        return best_id
    return None


def _window_records(
    predictor,
    image_paths: list[Path],
    frame_ids_10hz: list[int],
    scene_frame_start: int,
    view_id: int,
    prompts: Iterable[str],
    stitched_tracks: Dict[int, dict],
    next_local_id: int,
    emit_overlap: bool,
    temp_root: Path,
) -> tuple[list[dict], int]:
    records: list[dict] = []
    with tempfile.TemporaryDirectory(dir=str(temp_root), prefix=f"sam3_view{view_id}_") as tmp:
        tmp_dir = Path(tmp)
        _write_view_symlinks(image_paths, tmp_dir)
        for prompt in prompts:
            prompt_outputs, refined_obj_ids = _start_and_propagate(predictor, tmp_dir, prompt)
            window_tracks: Dict[int, dict] = {}
            for frame_offset, frame_id in enumerate(frame_ids_10hz):
                output = prompt_outputs.get(frame_offset, {})
                masks, boxes, scores, obj_ids = _masks_from_output(output)
                for obj_idx in range(masks.shape[0]):
                    temp_id = int(obj_ids[obj_idx])
                    track = window_tracks.setdefault(
                        temp_id,
                        {
                            "class_name": prompt,
                            "frames": {},
                            "boxes": {},
                            "scores": [],
                            "refined": bool(temp_id in refined_obj_ids),
                        },
                    )
                    track["refined"] = bool(track["refined"] or temp_id in refined_obj_ids)
                    track["frames"][scene_frame_start + frame_offset] = masks[obj_idx]
                    track["boxes"][scene_frame_start + frame_offset] = _xywh_to_xyxy(boxes[obj_idx]) if obj_idx < len(boxes) else [0, 0, 0, 0]
                    track["scores"].append(float(scores[obj_idx]) if obj_idx < len(scores) else 0.0)

            overlap_start = scene_frame_start
            for _, track in sorted(window_tracks.items(), key=lambda item: item[0]):
                assigned = _match_window_track(prompt, track["frames"], stitched_tracks, overlap_start)
                if assigned is None:
                    assigned = next_local_id
                    next_local_id += 1
                    stitched_tracks[assigned] = {"class_name": prompt, "frames": {}}
                stitched_tracks[assigned]["frames"].update(track["frames"])
                mean_score = float(np.mean(track["scores"])) if track["scores"] else 0.0
                for global_frame_idx, mask in sorted(track["frames"].items()):
                    if (not emit_overlap) and global_frame_idx < scene_frame_start:
                        continue
                    frame_offset = global_frame_idx - scene_frame_start
                    if frame_offset < 0 or frame_offset >= len(frame_ids_10hz):
                        continue
                    quality_ok, box_like_flag, quality_reason = _quality_flags(mask, refined=bool(track["refined"]))
                    bbox_tight = mask_tight_bbox_xyxy(mask)
                    records.append(
                        {
                            "record_id": f"{view_id}:{global_frame_idx}:{assigned}",
                            "scene_frame_index": global_frame_idx,
                            "frame_id_10hz": int(frame_ids_10hz[frame_offset]),
                            "view_id": int(view_id),
                            "local_mask_id": int(assigned),
                            "class_name": prompt,
                            "score": mean_score,
                            "prompt_type": "text",
                            "bbox_mask_tight": bbox_tight,
                            "visible_area": int(mask.sum()),
                            "rle": binary_mask_to_rle(mask),
                            "refined": bool(track["refined"]),
                            "quality_ok": bool(quality_ok),
                            "box_like_flag": bool(box_like_flag),
                            "quality_reason": quality_reason,
                            "global_weak_id": -1,
                            "matched_track_id": -1,
                            "match_score": 0.0,
                            "matched_proj_box_xyxy": None,
                            "matched_proj_box_xywh": None,
                        }
                    )
    return records, next_local_id


def _dedupe_records(records: list[dict]) -> list[dict]:
    unique: Dict[str, dict] = {}
    for record in records:
        unique[record["record_id"]] = record
    return [unique[key] for key in sorted(unique, key=lambda item: tuple(int(v) for v in item.split(":")))]


def _load_instances_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _restore_stitched_tracks(records: list[dict]) -> tuple[Dict[int, dict], int]:
    stitched_tracks: Dict[int, dict] = {}
    next_local_id = 1
    for record in records:
        local_id = int(record["local_mask_id"])
        stitched_tracks.setdefault(
            local_id,
            {"class_name": record["class_name"], "frames": {}},
        )["frames"][int(record["scene_frame_index"])] = rle_to_binary_mask(record["rle"])
        next_local_id = max(next_local_id, local_id + 1)
    return stitched_tracks, next_local_id


def _load_progress(path: Path) -> dict:
    if not path.exists():
        return {"completed_windows": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _instances_jsonl_complete(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            saw_payload = False
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    return False
                required = {"record_id", "scene_frame_index", "frame_id_10hz", "view_id", "local_mask_id", "rle", "quality_ok"}
                if not required.issubset(record):
                    return False
                saw_payload = True
        return saw_payload
    except Exception:
        return False


def _write_instances_jsonl_atomic(path: Path, records: list[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _build_frames_meta(records: list[dict], frame_ids_10hz: list[int], view_id: int) -> list[dict]:
    by_frame: Dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_frame[int(record["frame_id_10hz"])].append(record)
    frames_meta: list[dict] = []
    for scene_frame_index, frame_id in enumerate(frame_ids_10hz):
        frame_records = by_frame.get(int(frame_id), [])
        quality_masks = []
        rejected = 0
        for record in frame_records:
            if bool(record.get("quality_ok", False)) and (not bool(record.get("box_like_flag", False))):
                quality_masks.append(rle_to_binary_mask(record["rle"]))
            else:
                rejected += 1
        exhaustive_ok = bool(quality_masks) and rejected == 0
        valid_region_rle = None
        if (not exhaustive_ok) and quality_masks:
            union = np.logical_or.reduce(quality_masks)
            valid_region_rle = binary_mask_to_rle(union)
        frames_meta.append(
            {
                "scene_frame_index": int(scene_frame_index),
                "frame_id_10hz": int(frame_id),
                "view_id": int(view_id),
                "exhaustive_ok": bool(exhaustive_ok),
                "valid_region_rle": valid_region_rle,
                "quality_ok_count": int(len(quality_masks)),
                "rejected_count": int(rejected),
            }
        )
    return frames_meta


def _write_frames_meta_jsonl(path: Path, records: list[dict], frame_ids_10hz: list[int], view_id: int) -> None:
    frames_meta = _build_frames_meta(records, frame_ids_10hz, view_id)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in frames_meta:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _assign_global_weak_ids(records: list[dict], scene_dir: Path, scene_frame_ids: list[int], view_id: int) -> None:
    if not records:
        return
    by_frame: Dict[int, list[dict]] = defaultdict(list)
    decoded_masks: Dict[str, np.ndarray] = {}
    for record in records:
        by_frame[int(record["scene_frame_index"])].append(record)
        decoded_masks[record["record_id"]] = None
    track_arrays = build_track_arrays(scene_dir, scene_frame_ids)
    if track_arrays["track_boxes_3d"].shape[1] == 0:
        return
    first_pose = np.loadtxt(scene_dir / "lidar_pose" / f"{scene_frame_ids[0]:03d}.txt", dtype=np.float64).reshape(4, 4)
    for scene_frame_index, frame_records in by_frame.items():
        candidates = []
        for obj_idx in np.flatnonzero(track_arrays["track_visible"][scene_frame_index]):
            proj_box = _project_box_xyxy(
                scene_dir,
                scene_frame_ids[scene_frame_index],
                view_id,
                first_pose,
                track_arrays["track_boxes_3d"][scene_frame_index, obj_idx],
            )
            if proj_box is not None:
                candidates.append((obj_idx, proj_box))
        if not candidates:
            continue
        pair_scores = []
        for rec_idx, record in enumerate(frame_records):
            if decoded_masks[record["record_id"]] is None:
                decoded_masks[record["record_id"]] = rle_to_binary_mask(record["rle"])
            mask = decoded_masks[record["record_id"]]
            for obj_idx, proj_box in candidates:
                bbox_score = bbox_iou_xyxy(record["bbox_mask_tight"], proj_box)
                centroid_score = 1.0 if centroid_in_bbox(mask, proj_box) else 0.0
                class_score = _track_box_compatibility(record["class_name"], int(track_arrays["track_cls"][scene_frame_index, obj_idx]))
                score = 0.7 * bbox_score + 0.2 * centroid_score + 0.1 * class_score
                pair_scores.append((score, rec_idx, obj_idx))
        pair_scores.sort(reverse=True)
        used_records = set()
        used_tracks = set()
        for score, rec_idx, obj_idx in pair_scores:
            if score <= 0.4 or rec_idx in used_records or obj_idx in used_tracks:
                continue
            proj_box_xyxy = next(proj_box for cand_idx, proj_box in candidates if cand_idx == obj_idx)
            frame_records[rec_idx]["global_weak_id"] = int(track_arrays["track_id"][scene_frame_index, obj_idx])
            frame_records[rec_idx]["matched_track_id"] = int(track_arrays["track_id"][scene_frame_index, obj_idx])
            frame_records[rec_idx]["match_score"] = float(score)
            frame_records[rec_idx]["matched_proj_box_xyxy"] = [float(v) for v in proj_box_xyxy]
            frame_records[rec_idx]["matched_proj_box_xywh"] = xyxy_to_xywh(frame_records[rec_idx]["matched_proj_box_xyxy"])
            used_records.add(rec_idx)
            used_tracks.add(obj_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scene-level SAM3 and write primary instance assets.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--sam3-ckpt", type=Path, required=True)
    parser.add_argument("--prompts", nargs="*", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--start-scene-frame", type=int, default=0)
    parser.add_argument("--max-scene-frames", type=int, default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--view-ids", nargs="*", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--runtime-cache-root", type=Path, default=None)
    parser.add_argument("--no-rank-env", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = (args.output_root if args.output_root is not None else args.manifest.parent).resolve()
    shard_index, num_shards = distributed_shard_spec(
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        use_env=not args.no_rank_env,
    )
    temp_root = _configure_runtime_cache(
        output_root=output_root,
        shard_index=shard_index,
        runtime_cache_root=args.runtime_cache_root,
    )

    try:
        _preflight(args.sam3_root, args.sam3_ckpt)
    except FileNotFoundError as exc:
        raise SystemExit(f"[sam3 preflight] {exc}") from exc
    _install_sam3_import_shims()
    from sam3.model_builder import build_sam3_video_predictor

    entries = load_manifest(args.manifest)
    entries = filter_manifest_entries(entries, scene_ids=args.scene_ids)
    scene_entries = [entry for _, entry in manifest_scene_groups(entries)]
    if args.limit_scenes is not None:
        scene_entries = scene_entries[: args.limit_scenes]
    scene_entries = shard_items(scene_entries, shard_index=shard_index, num_shards=num_shards)

    predictor = build_sam3_video_predictor(
        checkpoint_path=str(args.sam3_ckpt),
        gpus_to_use=_predictor_gpu_ids_from_env(),
    )
    try:
        for entry in scene_entries:
            dirs = entry_output_dirs(entry, output_root)
            scene_dir = dirs["source_scene"]
            sam3_scene_dir = dirs["sam3_scene"]
            sam3_scene_dir.mkdir(parents=True, exist_ok=True)
            scene_view_ids = list(args.view_ids if args.view_ids is not None else entry["view_ids"])
            scene_frame_ids, scene_image_paths = collect_scene_image_paths(scene_dir, scene_view_ids)
            if args.start_scene_frame or args.max_scene_frames is not None:
                slice_start = max(int(args.start_scene_frame), 0)
                slice_end = len(scene_frame_ids) if args.max_scene_frames is None else min(len(scene_frame_ids), slice_start + int(args.max_scene_frames))
                scene_frame_ids = scene_frame_ids[slice_start:slice_end]
                scene_image_paths = {view_id: paths[slice_start:slice_end] for view_id, paths in scene_image_paths.items()}
            if not scene_frame_ids:
                print(f"[sam3_scene] skip {entry['scene_id']} no complete frames")
                continue
            track_arrays = build_track_arrays(scene_dir, scene_frame_ids)
            scene_meta = {
                "scene_id": entry["scene_id"],
                "frame_ids_10hz": scene_frame_ids,
                "view_ids": scene_view_ids,
                "window_len": args.window_len,
                "overlap": args.overlap,
                "start_scene_frame": args.start_scene_frame,
                "max_scene_frames": args.max_scene_frames,
                "prompts": list(args.prompts),
                "shard_index": shard_index,
                "num_shards": num_shards,
            }
            (sam3_scene_dir / "scene_meta.json").write_text(json.dumps(scene_meta, indent=2), encoding="utf-8")
            for view_id in scene_view_ids:
                out_path = sam3_scene_dir / f"view_{view_id}" / "instances.jsonl"
                frames_meta_path = sam3_scene_dir / f"view_{view_id}" / "frames_meta.jsonl"
                partial_path = sam3_scene_dir / f"view_{view_id}" / "instances.partial.jsonl"
                progress_path = sam3_scene_dir / f"view_{view_id}" / "progress.json"
                if _instances_jsonl_complete(out_path) and not args.overwrite:
                    print(f"[sam3_scene][{shard_index}/{num_shards}] skip {entry['scene_id']} view_{view_id}")
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                image_paths = scene_image_paths[int(view_id)]
                starts = _window_starts(len(scene_frame_ids), args.window_len, args.overlap)
                if args.overwrite:
                    partial_path.unlink(missing_ok=True)
                    progress_path.unlink(missing_ok=True)
                progress = _load_progress(progress_path)
                completed_windows = min(int(progress.get("completed_windows", 0)), len(starts))
                all_records = _load_instances_jsonl(partial_path)
                stitched_tracks, next_local_id = _restore_stitched_tracks(all_records)
                if completed_windows > 0:
                    print(
                        f"[sam3_scene][{shard_index}/{num_shards}] resume {entry['scene_id']} "
                        f"view_{view_id} from window {completed_windows + 1}/{len(starts)} "
                        f"partial_records={len(all_records)}"
                    )
                for win_idx, start in enumerate(starts[completed_windows:], start=completed_windows):
                    end = len(scene_frame_ids) if args.window_len <= 0 else min(start + args.window_len, len(scene_frame_ids))
                    emit_start = start if win_idx == 0 else start + args.overlap
                    records, next_local_id = _window_records(
                        predictor=predictor,
                        image_paths=image_paths[start:end],
                        frame_ids_10hz=scene_frame_ids[start:end],
                        scene_frame_start=start,
                        view_id=int(view_id),
                        prompts=args.prompts,
                        stitched_tracks=stitched_tracks,
                        next_local_id=next_local_id,
                        emit_overlap=(emit_start == start),
                        temp_root=temp_root,
                    )
                    if emit_start > start:
                        records = [record for record in records if record["scene_frame_index"] >= emit_start]
                    all_records.extend(records)
                    all_records = _dedupe_records(all_records)
                    _write_instances_jsonl_atomic(partial_path, all_records)
                    _write_progress(
                        progress_path,
                        {
                            "scene_id": entry["scene_id"],
                            "view_id": int(view_id),
                            "completed_windows": win_idx + 1,
                            "num_windows": len(starts),
                            "window_len": args.window_len,
                            "overlap": args.overlap,
                            "num_records_partial": len(all_records),
                            "status": "in_progress" if win_idx + 1 < len(starts) else "linking",
                        },
                    )
                    print(
                        f"[sam3_scene][{shard_index}/{num_shards}] checkpoint {entry['scene_id']} "
                        f"view_{view_id} window={win_idx + 1}/{len(starts)} records={len(all_records)}"
                    )
                all_records = _dedupe_records(all_records)
                _assign_global_weak_ids(all_records, scene_dir, scene_frame_ids, int(view_id))
                _write_instances_jsonl_atomic(out_path, all_records)
                _write_frames_meta_jsonl(frames_meta_path, all_records, scene_frame_ids, int(view_id))
                partial_path.unlink(missing_ok=True)
                _write_progress(
                    progress_path,
                    {
                        "scene_id": entry["scene_id"],
                        "view_id": int(view_id),
                        "completed_windows": len(starts),
                        "num_windows": len(starts),
                        "window_len": args.window_len,
                        "overlap": args.overlap,
                        "num_records_final": len(all_records),
                        "status": "done",
                    },
                )
                quality_ok = sum(int(record["quality_ok"]) for record in all_records)
                print(
                    f"[sam3_scene][{shard_index}/{num_shards}] {entry['scene_id']} view_{view_id} "
                    f"records={len(all_records)} quality_ok={quality_ok} "
                    f"tracks={track_arrays['track_boxes_3d'].shape[1]}"
                )
    finally:
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()


if __name__ == "__main__":
    main()
