"""Batch assembly helpers for stage-B overfit training."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
from PIL import Image

from dvgt.datasets.transforms.transforms import (
    crop_image_depth_and_intrinsic_by_pp,
    resize_image_depth_and_intrinsic,
)
from tools.data_process.utils.image import process_image_sequentially, resize_to_aspect_ratio


DEFAULT_CACHE_KEYS = (
    "points.npy",
    "points_conf.npy",
    "ego_pose.npy",
    "raw_patch_tokens.npy",
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

_RFU_FROM_RDF = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_RDF_FROM_RFU = _RFU_FROM_RDF.T.copy()


def _rfu_pose_to_rdf(pose_rfu: np.ndarray) -> np.ndarray:
    """Convert transforms whose input and output ego axes are RFU into DVGT RDF."""
    return (_RDF_FROM_RFU @ pose_rfu @ _RFU_FROM_RDF).astype(np.float32, copy=False)


def _rfu_output_to_rdf(transform: np.ndarray) -> np.ndarray:
    """Convert only the output ego axes from RFU to RDF; camera input stays OpenCV/RDF."""
    return (_RDF_FROM_RFU @ transform).astype(np.float32, copy=False)


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


def _intr4_to_matrix(intr: np.ndarray) -> np.ndarray:
    flat = np.asarray(intr, dtype=np.float32).reshape(-1)
    if flat.size >= 9 and abs(float(flat[8])) > 1e-6:
        return flat[:9].reshape(3, 3).astype(np.float32, copy=True)
    if flat.size < 4:
        raise ValueError(f"Expected at least fx/fy/cx/cy intrinsics, got shape {flat.shape}.")
    matrix = np.eye(3, dtype=np.float32)
    matrix[0, 0] = flat[0]
    matrix[1, 1] = flat[1]
    matrix[0, 2] = flat[2]
    matrix[1, 2] = flat[3]
    return matrix


def _matrix_to_intr4(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.asarray([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]], dtype=np.float32)


def _dvgt_preprocess_image_and_intrinsics(
    path: str,
    intr: np.ndarray,
    size_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(path).convert("RGB"))
    original_aspect_ratio = float(image.shape[1]) / max(float(image.shape[0]), 1.0)
    target_shape = np.asarray([int(size_hw[0]), int(size_hw[1])], dtype=np.int32)
    intr_matrix = _intr4_to_matrix(intr)

    image, intr_matrix, _ = process_image_sequentially(image, intr_matrix, num_tokens=2500)
    image, intr_matrix = resize_to_aspect_ratio(image, intr_matrix, original_aspect_ratio)

    original_size = np.asarray(image.shape[:2], dtype=np.int32)
    image, _, intr_matrix, _ = crop_image_depth_and_intrinsic_by_pp(
        image,
        None,
        intr_matrix,
        original_size,
        track=None,
        filepath=str(path),
        strict=False,
    )
    image, _, intr_matrix, _ = resize_image_depth_and_intrinsic(
        image,
        None,
        intr_matrix,
        target_shape,
        track=None,
        enable_random_resize=False,
    )
    image, _, intr_matrix, _ = crop_image_depth_and_intrinsic_by_pp(
        image,
        None,
        intr_matrix,
        target_shape,
        track=None,
        filepath=str(path),
        strict=True,
    )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)), _matrix_to_intr4(intr_matrix)


def _load_view_intrinsics(scene_path: str, view_id: int) -> np.ndarray:
    return np.loadtxt(f"{scene_path}/intrinsics/{int(view_id)}.txt", dtype=np.float32).reshape(-1)


def _load_rgb_clip(
    image_paths: Mapping[int, List[str]],
    view_ids: List[int],
    size_hw: tuple[int, int],
    *,
    scene_path: str | None = None,
) -> np.ndarray:
    h, w = int(size_hw[0]), int(size_hw[1])
    frames = len(next(iter(image_paths.values())))
    rgb = np.zeros((frames, len(view_ids), 3, h, w), dtype=np.float32)
    for v_idx, view_id in enumerate(view_ids):
        paths = image_paths.get(view_id) or image_paths.get(str(view_id))
        intr = _load_view_intrinsics(scene_path, int(view_id)) if scene_path is not None else None
        for t_idx, path in enumerate(paths):
            if intr is not None:
                rgb[t_idx, v_idx], _ = _dvgt_preprocess_image_and_intrinsics(str(path), intr, size_hw)
            else:
                image = Image.open(path).convert("RGB")
                image = image.resize((w, h), Image.Resampling.BICUBIC)
                arr = np.asarray(image, dtype=np.float32) / 255.0
                rgb[t_idx, v_idx] = np.transpose(arr, (2, 0, 1))
    return rgb


def _load_camera_params(
    scene_dir: str,
    frame_ids: List[int],
    view_ids: List[int],
    *,
    image_paths: Mapping[int, List[str]],
    target_size_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scene_path = str(scene_dir)
    intrinsics = np.zeros((len(view_ids), 4), dtype=np.float32)
    camera_to_world = np.zeros((len(frame_ids), len(view_ids), 4, 4), dtype=np.float32)
    first_ego_pose = np.loadtxt(f"{scene_path}/lidar_pose/{int(frame_ids[0]):03d}.txt", dtype=np.float32).reshape(4, 4)
    first_world_to_ego = np.linalg.inv(first_ego_pose).astype(np.float32)
    ego_n_to_first_ego = np.zeros((len(frame_ids), 4, 4), dtype=np.float32)
    for t_idx, frame_id in enumerate(frame_ids):
        ego_to_world = np.loadtxt(f"{scene_path}/lidar_pose/{int(frame_id):03d}.txt", dtype=np.float32).reshape(4, 4)
        ego_n_to_first_ego[t_idx] = first_world_to_ego @ ego_to_world
    target_h, target_w = int(target_size_hw[0]), int(target_size_hw[1])
    for v_idx, view_id in enumerate(view_ids):
        intr = _load_view_intrinsics(scene_path, int(view_id))
        intr_scaled = intr[:4].astype(np.float32, copy=True)
        paths = image_paths.get(view_id) or image_paths.get(str(view_id))
        if paths:
            with Image.open(paths[0]) as image:
                orig_w, orig_h = image.size
            if orig_w > 0 and orig_h > 0:
                intr_scaled[[0, 2]] *= float(target_w) / float(orig_w)
                intr_scaled[[1, 3]] *= float(target_h) / float(orig_h)
        intrinsics[v_idx] = intr_scaled
        for t_idx, frame_id in enumerate(frame_ids):
            camera_to_world[t_idx, v_idx] = np.loadtxt(
                f"{scene_path}/extrinsics/{int(frame_id):03d}_{int(view_id)}.txt",
                dtype=np.float32,
            ).reshape(4, 4)
    return intrinsics, camera_to_world, first_ego_pose, ego_n_to_first_ego


def _load_intrinsics_scaled(
    scene_path: str,
    view_ids: List[int],
    *,
    image_paths: Mapping[int, List[str]],
    target_size_hw: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = int(target_size_hw[0]), int(target_size_hw[1])
    intrinsics = np.zeros((len(view_ids), 4), dtype=np.float32)
    for v_idx, view_id in enumerate(view_ids):
        intr = _load_view_intrinsics(scene_path, int(view_id))
        intr_scaled = intr[:4].astype(np.float32, copy=True)
        paths = image_paths.get(view_id) or image_paths.get(str(view_id))
        if paths:
            _, intr_scaled = _dvgt_preprocess_image_and_intrinsics(str(paths[0]), intr, (target_h, target_w))
        intrinsics[v_idx] = intr_scaled
    return intrinsics


def _load_online_camera_params(
    scene_dir: str,
    frame_ids: List[int],
    source_view_ids: List[int],
    target_view_ids: List[int],
    *,
    source_image_paths: Mapping[int, List[str]],
    target_image_paths: Mapping[int, List[str]],
    target_size_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scene_path = str(scene_dir)
    first_ego_pose = np.loadtxt(f"{scene_path}/lidar_pose/{int(frame_ids[0]):03d}.txt", dtype=np.float32).reshape(4, 4)
    first_world_to_ego = np.linalg.inv(first_ego_pose).astype(np.float32)
    ego_n_to_first_ego = np.zeros((len(frame_ids), 4, 4), dtype=np.float32)
    for t_idx, frame_id in enumerate(frame_ids):
        ego_to_world = np.loadtxt(f"{scene_path}/lidar_pose/{int(frame_id):03d}.txt", dtype=np.float32).reshape(4, 4)
        ego_n_to_first_ego[t_idx] = first_world_to_ego @ ego_to_world
    ego_n_to_first_ego = _rfu_pose_to_rdf(ego_n_to_first_ego)

    def camera_to_ego_for(view_ids: List[int]) -> np.ndarray:
        transforms = np.zeros((len(view_ids), 4, 4), dtype=np.float32)
        for v_idx, view_id in enumerate(view_ids):
            camera_to_world = np.loadtxt(
                f"{scene_path}/extrinsics/{int(frame_ids[0]):03d}_{int(view_id)}.txt",
                dtype=np.float32,
            ).reshape(4, 4)
            transforms[v_idx] = _rfu_output_to_rdf(first_world_to_ego @ camera_to_world)
        return transforms

    source_intrinsics = _load_intrinsics_scaled(
        scene_path,
        source_view_ids,
        image_paths=source_image_paths,
        target_size_hw=target_size_hw,
    )
    target_intrinsics = _load_intrinsics_scaled(
        scene_path,
        target_view_ids,
        image_paths=target_image_paths,
        target_size_hw=target_size_hw,
    )
    source_camera_to_ego = camera_to_ego_for(source_view_ids)
    target_camera_to_ego = camera_to_ego_for(target_view_ids)
    gt_source_camera_to_first = np.einsum("tij,vjk->tvik", ego_n_to_first_ego, source_camera_to_ego, optimize=True)
    gt_target_camera_to_first = np.einsum("tij,vjk->tvik", ego_n_to_first_ego, target_camera_to_ego, optimize=True)
    return (
        source_intrinsics,
        target_intrinsics,
        source_camera_to_ego,
        target_camera_to_ego,
        first_ego_pose,
        ego_n_to_first_ego,
        gt_source_camera_to_first,
        gt_target_camera_to_first,
    )


def _ego_points_to_first_ego(
    points_ego_n: np.ndarray,
    ego_n_to_first_ego: np.ndarray,
) -> np.ndarray:
    """Convert cached DVGT point-head output from per-frame ego_n to first ego."""
    flat = points_ego_n.astype(np.float32, copy=False).reshape(*points_ego_n.shape[:2], -1, 3)
    ego_n_to_first_ego = ego_n_to_first_ego.astype(np.float32, copy=False)
    rot = ego_n_to_first_ego[..., :3, :3]
    trans = ego_n_to_first_ego[..., :3, 3]
    flat_first = np.einsum("tvnc,tdc->tvnd", flat, rot, optimize=True) + trans[:, None, None, :]
    return flat_first.reshape(points_ego_n.shape).astype(np.float32, copy=False)


def collate_dvgt_occ_batch(samples: List[Mapping[str, object]]) -> Dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    if "dvgt_cache" not in samples[0]:
        return _collate_online_dvgt_batch(samples)

    cache = [sample["dvgt_cache"] for sample in samples]
    supervision = [sample["supervision"] for sample in samples]

    batch: Dict[str, object] = {
        "scene_id": [str(sample["scene_id"]) for sample in samples],
        "clip_id": [str(sample["clip_id"]) for sample in samples],
        "frame_ids_10hz": [sample["frame_ids_10hz"] for sample in samples],
        "view_ids": [sample["view_ids"] for sample in samples],
        "image_paths": [sample["image_paths"] for sample in samples],
        "source_scene_dir": [str(sample["source_scene_dir"]) for sample in samples],
        "aggregated_tokens": {
            4: _stack_fixed([item["feat_l4"] for item in cache]),
            11: _stack_fixed([item["feat_l11"] for item in cache]),
            17: _stack_fixed([item["feat_l17"] for item in cache]),
            23: _stack_fixed([item["feat_l23"] for item in cache]),
        },
        "raw_patch_tokens": _stack_fixed([item["raw_patch_tokens"] for item in cache]),
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
        batch["sam3_valid_mask_full"] = _stack_fixed([item["sam3_valid_mask_full"] for item in sam3_full]).float()
        if "sky_mask_full" in sam3_full[0]:
            batch["sky_mask_full"] = _stack_fixed([item["sky_mask_full"] for item in sam3_full]).float()
    size_hw = tuple(int(dim) for dim in batch["points"].shape[3:5])
    batch["rgb_target"] = _stack_fixed(
        [
            _load_rgb_clip(sample["image_paths"], list(sample["view_ids"]), size_hw)
            for sample in samples
        ]
    ).float()
    intrinsics, camera_to_world, first_ego_pose, ego_n_to_first_ego = zip(
        *[
            _load_camera_params(
                str(sample["source_scene_dir"]),
                list(sample["frame_ids_10hz"]),
                list(sample["view_ids"]),
                image_paths=sample["image_paths"],
                target_size_hw=size_hw,
            )
            for sample in samples
        ]
    )
    points_first_ego = [
        _ego_points_to_first_ego(item["points"], ego_to_first)
        for item, ego_to_first in zip(cache, ego_n_to_first_ego)
    ]
    batch["points"] = _stack_fixed(points_first_ego).float()
    batch["camera_intrinsics"] = _stack_fixed(list(intrinsics)).float()
    batch["camera_to_world"] = _stack_fixed(list(camera_to_world)).float()
    batch["first_ego_pose_world"] = _stack_fixed(list(first_ego_pose)).float()
    return batch


def _collate_online_dvgt_batch(samples: List[Mapping[str, object]]) -> Dict[str, object]:
    supervision = [sample["supervision"] for sample in samples]
    size_hw = tuple(int(dim) for dim in samples[0].get("full_res_size", (224, 448)))
    batch: Dict[str, object] = {
        "online_dvgt": True,
        "scene_id": [str(sample["scene_id"]) for sample in samples],
        "clip_id": [str(sample["clip_id"]) for sample in samples],
        "frame_ids_10hz": [sample["target_frame_ids_10hz"] for sample in samples],
        "source_frame_ids_10hz": [sample["source_frame_ids_10hz"] for sample in samples],
        "target_frame_ids_10hz": [sample["target_frame_ids_10hz"] for sample in samples],
        "view_ids": [sample["target_view_ids"] for sample in samples],
        "source_view_ids": [sample["source_view_ids"] for sample in samples],
        "target_view_ids": [sample["target_view_ids"] for sample in samples],
        "image_paths": [sample["target_image_paths"] for sample in samples],
        "source_image_paths": [sample["source_image_paths"] for sample in samples],
        "target_image_paths": [sample["target_image_paths"] for sample in samples],
        "source_scene_dir": [str(sample["source_scene_dir"]) for sample in samples],
        "online_sample_meta": [sample.get("online_sample_meta", {}) for sample in samples],
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
    batch["source_rgb"] = _stack_fixed(
        [
            _load_rgb_clip(
                sample["source_image_paths"],
                list(sample["source_view_ids"]),
                size_hw,
                scene_path=str(sample["source_scene_dir"]),
            )
            for sample in samples
        ]
    ).float()
    batch["rgb_target"] = _stack_fixed(
        [
            _load_rgb_clip(
                sample["target_image_paths"],
                list(sample["target_view_ids"]),
                size_hw,
                scene_path=str(sample["source_scene_dir"]),
            )
            for sample in samples
        ]
    ).float()
    target_is_source = []
    for sample in samples:
        source_set = {int(view_id) for view_id in sample["source_view_ids"]}
        target_is_source.append(
            np.asarray([int(view_id) in source_set for view_id in sample["target_view_ids"]], dtype=np.bool_)
        )
    batch["target_is_source"] = _stack_fixed(target_is_source).bool()
    batch["target_is_holdout"] = ~batch["target_is_source"]
    if "sam3_fullres" in samples[0]:
        sam3_full = [sample["sam3_fullres"] for sample in samples]
        batch["sam3_dyn_mask_full"] = _stack_fixed([item["sam3_dyn_mask_full"] for item in sam3_full]).float()
        batch["sam3_all_mask_full"] = _stack_fixed([item["sam3_all_mask_full"] for item in sam3_full]).float()
        batch["sam3_semantic_full"] = _stack_fixed([item["sam3_semantic_full"] for item in sam3_full]).float()
        batch["sam3_valid_mask_full"] = _stack_fixed([item["sam3_valid_mask_full"] for item in sam3_full]).float()
        if "sky_mask_full" in sam3_full[0]:
            batch["sky_mask_full"] = _stack_fixed([item["sky_mask_full"] for item in sam3_full]).float()
    params = [
        _load_online_camera_params(
            str(sample["source_scene_dir"]),
            list(sample["source_frame_ids_10hz"]),
            list(sample["source_view_ids"]),
            list(sample["target_view_ids"]),
            source_image_paths=sample["source_image_paths"],
            target_image_paths=sample["target_image_paths"],
            target_size_hw=size_hw,
        )
        for sample in samples
    ]
    (
        source_intrinsics,
        target_intrinsics,
        source_camera_to_ego,
        target_camera_to_ego,
        first_ego_pose,
        ego_n_to_first_ego,
        gt_source_camera_to_first,
        gt_target_camera_to_first,
    ) = zip(*params)
    batch["source_camera_intrinsics"] = _stack_fixed(list(source_intrinsics)).float()
    batch["target_camera_intrinsics"] = _stack_fixed(list(target_intrinsics)).float()
    batch["camera_intrinsics"] = batch["target_camera_intrinsics"]
    batch["source_camera_to_ego"] = _stack_fixed(list(source_camera_to_ego)).float()
    batch["target_camera_to_ego"] = _stack_fixed(list(target_camera_to_ego)).float()
    batch["gt_first_ego_pose_world"] = _stack_fixed(list(first_ego_pose)).float()
    batch["gt_ego_n_to_first_ego"] = _stack_fixed(list(ego_n_to_first_ego)).float()
    batch["gt_source_camera_to_first_ego"] = _stack_fixed(list(gt_source_camera_to_first)).float()
    batch["gt_target_camera_to_first_ego"] = _stack_fixed(list(gt_target_camera_to_first)).float()
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
