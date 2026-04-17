"""Batch assembly helpers for stage-B overfit training."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch


DEFAULT_CACHE_KEYS = (
    "points.npy",
    "points_conf.npy",
    "ego_pose.npy",
    "feat_l4.npy",
    "feat_l11.npy",
    "feat_l17.npy",
    "feat_l23.npy",
)

DEFAULT_SUPERVISION_KEYS = (
    "dyn_soft_mask_1_4.npy",
    "gs_local_inst_id_1_8.npy",
    "gs_global_track_id_1_8.npy",
    "inst_boundary_weight_1_8.npy",
    "inst_area_ratio_1_8.npy",
    "inst_quality_mask_1_8.npy",
    "track_boxes_3d.npy",
    "track_cls.npy",
    "track_id.npy",
    "track_visible.npy",
    "track_birth.npy",
    "track_death.npy",
    "track_delta_box.npy",
    "occ_label.npy",
    "occ_sem_label.npy",
    "occ_dyn_label.npy",
)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    if tensor.dtype == torch.uint8:
        return tensor.long()
    return tensor


def _stack_fixed(arrays: Iterable[np.ndarray]) -> torch.Tensor:
    return torch.stack([_to_tensor(array) for array in arrays], dim=0)


def _pad_track_tensor(arrays: List[np.ndarray], pad_value: int | float | bool) -> torch.Tensor:
    max_tracks = max(array.shape[1] for array in arrays)
    padded = []
    for array in arrays:
        pad_shape = list(array.shape)
        pad_shape[1] = max_tracks
        filled = np.full(pad_shape, pad_value, dtype=array.dtype)
        filled[:, : array.shape[1], ...] = array
        padded.append(_to_tensor(filled))
    return torch.stack(padded, dim=0)


def collate_dvgt_occ_batch(samples: List[Mapping[str, object]]) -> Dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    cache = [sample["dvgt_cache"] for sample in samples]
    supervision = [sample["supervision"] for sample in samples]

    batch: Dict[str, object] = {
        "scene_id": [str(sample["scene_id"]) for sample in samples],
        "clip_id": [str(sample["clip_id"]) for sample in samples],
        "frame_ids_10hz": [sample["frame_ids_10hz"] for sample in samples],
        "view_ids": [sample["view_ids"] for sample in samples],
        "aggregated_tokens": {
            4: _stack_fixed([item["feat_l4"] for item in cache]),
            11: _stack_fixed([item["feat_l11"] for item in cache]),
            17: _stack_fixed([item["feat_l17"] for item in cache]),
            23: _stack_fixed([item["feat_l23"] for item in cache]),
        },
        "points": _stack_fixed([item["points"] for item in cache]),
        "points_conf": _stack_fixed([item["points_conf"] for item in cache]),
        "ego_pose": _stack_fixed([item["ego_pose"] for item in cache]),
        "dyn_soft_mask_1_4": _stack_fixed([item["dyn_soft_mask_1_4"] for item in supervision]).float(),
        "gs_local_inst_id_1_8": _stack_fixed([item["gs_local_inst_id_1_8"] for item in supervision]).long(),
        "gs_global_track_id_1_8": _stack_fixed([item["gs_global_track_id_1_8"] for item in supervision]).long(),
        "inst_boundary_weight_1_8": _stack_fixed([item["inst_boundary_weight_1_8"] for item in supervision]).float(),
        "inst_area_ratio_1_8": _stack_fixed([item["inst_area_ratio_1_8"] for item in supervision]).float(),
        "inst_quality_mask_1_8": _stack_fixed([item["inst_quality_mask_1_8"] for item in supervision]).float(),
        "occ_label": _stack_fixed([item["occ_label"] for item in supervision]).float(),
        "occ_sem_label": _stack_fixed([item["occ_sem_label"] for item in supervision]).long(),
        "occ_dyn_label": _stack_fixed([item["occ_dyn_label"] for item in supervision]).float(),
        "track_boxes_3d": _pad_track_tensor([item["track_boxes_3d"] for item in supervision], 0.0).float(),
        "track_cls": _pad_track_tensor([item["track_cls"] for item in supervision], -1).long(),
        "track_id": _pad_track_tensor([item["track_id"] for item in supervision], -1).long(),
        "track_visible": _pad_track_tensor([item["track_visible"] for item in supervision], False).bool(),
        "track_birth": _pad_track_tensor([item["track_birth"] for item in supervision], False).bool(),
        "track_death": _pad_track_tensor([item["track_death"] for item in supervision], False).bool(),
        "track_delta_box": _pad_track_tensor([item["track_delta_box"] for item in supervision], 0.0).float(),
    }
    if "sam3_fullres" in samples[0]:
        sam3_full = [sample["sam3_fullres"] for sample in samples]
        batch["sam3_dyn_mask_full"] = _stack_fixed([item["sam3_dyn_mask_full"] for item in sam3_full]).float()
        batch["sam3_all_mask_full"] = _stack_fixed([item["sam3_all_mask_full"] for item in sam3_full]).float()
        batch["sam3_semantic_full"] = _stack_fixed([item["sam3_semantic_full"] for item in sam3_full]).float()
    return batch


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            moved[key] = {
                inner_key: inner_value.to(device, non_blocking=True) if isinstance(inner_value, torch.Tensor) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            moved[key] = value
    return moved
