#!/usr/bin/env python3
"""Export standalone qualitative assets for thesis pipeline diagrams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ import DEFAULT_DVGT_OCC_CONFIG
from dvgt_occ.data import DVGTOccClipDataset
from dvgt_occ.data.stage_a_utils import load_lidar_points, load_lidar_pose, rle_to_binary_mask, transform_points


SEMANTIC_PALETTE = np.array(
    [
        [220, 20, 60],
        [255, 140, 0],
        [255, 215, 0],
        [199, 21, 133],
        [160, 82, 45],
        [30, 144, 255],
        [50, 205, 50],
        [148, 0, 211],
        [0, 206, 209],
    ],
    dtype=np.float32,
) / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export multi-panel qualitative assets for the DVGT-Occ pipeline.")
    parser.add_argument("--manifest", type=Path, default=Path("data/nuscenes_dvgt_v0/manifest_trainval.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/nuscenes_dvgt_v0"))
    parser.add_argument("--scene-id", type=str, default="000")
    parser.add_argument("--clip-id", type=str, default="clip_000000")
    parser.add_argument("--frames", type=int, nargs="*", default=[0, 3, 7])
    parser.add_argument("--views", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--semantic-view", type=int, default=0)
    parser.add_argument("--semantic-frame", type=int, default=0)
    parser.add_argument("--voxel-frame", type=int, default=0)
    parser.add_argument("--point-frame", type=int, default=-1, help="-1 means aggregate all frames.")
    parser.add_argument("--max-pointcloud-points", type=int, default=50000)
    parser.add_argument("--sam3-attention-npy", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_sample(args: argparse.Namespace) -> dict:
    dataset = DVGTOccClipDataset(
        manifest_path=args.manifest,
        root=args.output_root,
        load_cache=True,
        load_supervision=True,
        load_scene_sam3_full=True,
        scene_ids=[args.scene_id],
        clip_ids=[args.clip_id],
        limit=1,
        full_res_size=(DEFAULT_DVGT_OCC_CONFIG.image_height, DEFAULT_DVGT_OCC_CONFIG.image_width),
        projected_semantic_classes=DEFAULT_DVGT_OCC_CONFIG.projected_semantic_classes,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Could not find sample scene={args.scene_id} clip={args.clip_id}.")
    return dataset[0]


def load_rgb(path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((size_hw[1], size_hw[0]), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _subsample_xyz(xyz: np.ndarray, weight: np.ndarray, max_points: int) -> np.ndarray:
    xyz = xyz.reshape(-1, 3)
    weight = weight.reshape(-1)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(weight)
    xyz = xyz[finite]
    weight = weight[finite]
    if xyz.shape[0] == 0:
        return xyz
    keep = weight >= np.percentile(weight, 75.0)
    xyz = xyz[keep]
    if xyz.shape[0] == 0:
        return xyz
    depth = np.linalg.norm(xyz, axis=1)
    xyz = xyz[depth <= 80.0]
    if xyz.shape[0] > max_points:
        stride = max(int(np.ceil(xyz.shape[0] / float(max_points))), 1)
        xyz = xyz[::stride]
    return xyz


def save_rgb_grid(sample: dict, out_path: Path, frame_indices: Sequence[int], view_indices: Sequence[int]) -> None:
    size_hw = (DEFAULT_DVGT_OCC_CONFIG.image_height, DEFAULT_DVGT_OCC_CONFIG.image_width)
    rows = len(frame_indices)
    cols = len(view_indices)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.0 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.expand_dims(np.asarray(axes), axis=0)
    if cols == 1:
        axes = np.expand_dims(np.asarray(axes), axis=1)
    for r, frame_idx in enumerate(frame_indices):
        for c, view_idx in enumerate(view_indices):
            ax = axes[r, c]
            safe_view_idx = max(0, min(view_idx, len(sample["view_ids"]) - 1))
            view_id = sample["view_ids"][safe_view_idx]
            image_path = Path(sample["image_paths"][int(view_id)][max(0, min(frame_idx, len(sample["image_paths"][int(view_id)]) - 1))])
            ax.imshow(load_rgb(image_path, size_hw))
            ax.set_title(f"t={frame_idx} view={view_id}", fontsize=9)
            ax.axis("off")
    fig.suptitle("Multi-view Multi-time RGB Clip", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_pointcloud(sample: dict, out_path: Path, frame_idx: int, max_points: int) -> None:
    scene_dir = Path(sample["source_scene_dir"])
    frame_ids = list(sample["frame_ids_10hz"])
    use_frame_ids = frame_ids if frame_idx < 0 else [frame_ids[max(0, min(frame_idx, len(frame_ids) - 1))]]
    first_inv = np.linalg.inv(load_lidar_pose(scene_dir, frame_ids[0]))
    xyz_list = []
    for frame_id in use_frame_ids:
        lidar = load_lidar_points(scene_dir / "lidar" / f"{int(frame_id):03d}.bin")
        lidar_to_first = first_inv @ load_lidar_pose(scene_dir, int(frame_id))
        xyz_list.append(transform_points(lidar, lidar_to_first))
    xyz = np.concatenate(xyz_list, axis=0).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    keep = (xyz[:, 0] >= -40.0) & (xyz[:, 0] <= 40.0) & (xyz[:, 1] >= -30.0) & (xyz[:, 1] <= 30.0) & (xyz[:, 2] >= -3.0) & (xyz[:, 2] <= 8.0)
    xyz = xyz[keep]
    if xyz.shape[0] > max_points:
        stride = max(int(np.ceil(xyz.shape[0] / float(max_points))), 1)
        xyz = xyz[::stride]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.08, c="black", alpha=0.7, linewidths=0.0)
    ax.view_init(elev=18, azim=-98)
    ax.set_xlim(-22.0, 22.0)
    ax.set_ylim(-16.0, 16.0)
    ax.set_zlim(-2.5, 6.5)
    ax.set_title("Scene Point Cloud (fused LiDAR, ego-0)")
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _colorize_instance_map(id_map: np.ndarray) -> np.ndarray:
    colored = np.zeros(id_map.shape + (3,), dtype=np.float32)
    valid = id_map >= 0
    if not np.any(valid):
        return colored
    flat_ids = np.unique(id_map[valid])
    rng = np.random.default_rng(0)
    palette = {int(idx): rng.random(3).astype(np.float32) for idx in flat_ids}
    for idx, color in palette.items():
        colored[id_map == idx] = color
    return colored


def build_sam3_instance_map(sample: dict, frame_idx: int, view_idx: int) -> np.ndarray:
    clip_index = sample["sam3_clip_index"]
    scene_dir = Path(sample["sam3_scene_dir"])
    t_idx = max(0, min(frame_idx, len(clip_index["frame_ids_10hz"]) - 1))
    v_idx = max(0, min(view_idx, len(clip_index["view_ids"]) - 1))
    record_ids = clip_index["records_by_frame_view"].get(f"{t_idx}_{v_idx}", [])
    id_map = np.full((DEFAULT_DVGT_OCC_CONFIG.image_height, DEFAULT_DVGT_OCC_CONFIG.image_width), -1, dtype=np.int32)
    next_id = 0
    view_id = clip_index["view_ids"][v_idx]
    instances_path = scene_dir / f"view_{view_id}" / "instances.jsonl"
    if not instances_path.exists():
        return id_map
    records = {}
    with instances_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["record_id"])] = record
    for record_id in record_ids:
        record = records.get(str(record_id))
        if record is None:
            continue
        if (not bool(record.get("quality_ok", False))) or bool(record.get("box_like_flag", False)):
            continue
        mask = rle_to_binary_mask(record["rle"]).astype(bool)
        if mask.shape != id_map.shape:
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            mask_img = mask_img.resize((id_map.shape[1], id_map.shape[0]), Image.Resampling.NEAREST)
            mask = np.asarray(mask_img, dtype=np.uint8) > 0
        id_map[mask] = next_id
        next_id += 1
    return id_map


def save_sam3_masks(sample: dict, out_mask_path: Path, out_id_path: Path, frame_idx: int, view_idx: int, attention_path: Path | None = None) -> None:
    t_idx = max(0, min(frame_idx, sample["sam3_fullres"]["sam3_all_mask_full"].shape[0] - 1))
    v_idx = max(0, min(view_idx, sample["sam3_fullres"]["sam3_all_mask_full"].shape[1] - 1))
    mask = sample["sam3_fullres"]["sam3_all_mask_full"][t_idx, v_idx]
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("SAM3 Full Mask")
    ax.axis("off")
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_mask_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.imshow(_colorize_instance_map(build_sam3_instance_map(sample, frame_idx=t_idx, view_idx=v_idx)))
    ax.set_title("SAM3 Instance IDs")
    ax.axis("off")
    fig.savefig(out_id_path, dpi=200)
    plt.close(fig)

    if attention_path is not None and attention_path.exists():
        attention = np.load(attention_path)
        if attention.ndim == 3:
            attention = attention.mean(axis=0)
        fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
        ax.imshow(attention, cmap="magma")
        ax.set_title("SAM3 Attention Heatmap")
        ax.axis("off")
        fig.savefig(out_mask_path.parent / "03c_sam3_attention.png", dpi=200)
        plt.close(fig)


def _downsample_binary_xy(volume: np.ndarray, factor_xy: int = 2) -> np.ndarray:
    nz, ny, nx = volume.shape
    ny2 = ny // factor_xy
    nx2 = nx // factor_xy
    trimmed = volume[:, : ny2 * factor_xy, : nx2 * factor_xy]
    pooled = trimmed.reshape(nz, ny2, factor_xy, nx2, factor_xy).max(axis=(2, 4))
    return pooled.astype(bool)


def save_voxel_volume(sample: dict, out_path: Path, frame_idx: int) -> None:
    occ = sample["supervision"]["occ_label"]
    t_idx = max(0, min(frame_idx, occ.shape[0] - 1))
    volume = (occ[t_idx] > 0.5).astype(np.float32)
    dense = F.max_pool3d(torch.from_numpy(volume)[None, None], kernel_size=3, stride=1, padding=1).squeeze().numpy() > 0.0
    coords = np.argwhere(dense)
    if coords.shape[0] == 0:
        raise RuntimeError("Empty occupancy volume; cannot render voxel figure.")
    lo = np.floor(np.percentile(coords, 2.0, axis=0)).astype(int)
    hi = np.ceil(np.percentile(coords, 98.0, axis=0)).astype(int) + 1
    lo = np.maximum(lo - np.array([1, 2, 2]), 0)
    hi = np.minimum(hi + np.array([1, 2, 2]), np.array(dense.shape))
    cropped = dense[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    while max(cropped.shape[1], cropped.shape[2]) > 28:
        cropped = _downsample_binary_xy(cropped, factor_xy=2)
    voxels = cropped.transpose(2, 1, 0)
    x, y, z = np.indices(np.array(voxels.shape) + 1)
    z_norm = np.linspace(0.25, 1.0, voxels.shape[2], dtype=np.float32)[None, None, :, None]
    base = np.array([0.28, 0.60, 0.95, 0.10], dtype=np.float32)
    top = np.array([0.90, 0.96, 1.00, 0.72], dtype=np.float32)
    colors = base[None, None, None, :] * (1.0 - z_norm) + top[None, None, None, :] * z_norm
    colors = np.broadcast_to(colors, voxels.shape + (4,)).copy()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(x, y, z, voxels, facecolors=colors, edgecolor=(1, 1, 1, 0.01))
    ax.view_init(elev=23, azim=-58)
    ax.set_title("Occupancy Volume (stylized from same-frame occ)")
    ax.set_box_aspect(voxels.shape)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _load_scene_records(scene_dir: Path, view_id: int) -> dict:
    path = scene_dir / f"view_{view_id}" / "instances.jsonl"
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["record_id"])] = record
    return records


def _class_to_semantic_idx(class_name: str) -> int:
    name = class_name.strip().lower().replace("_", " ")
    if "construction" in name:
        return 4
    if "pedestrian" in name:
        return 5
    if "cyclist" in name or "bicyclist" in name or "motorcyclist" in name or "rider" in name:
        return 8
    if "bicycle" in name or name == "bike":
        return 6
    if "motorcycle" in name or "motorbike" in name:
        return 7
    if "trailer" in name:
        return 3
    if "bus" in name:
        return 2
    if "truck" in name:
        return 1
    if "car" in name or "vehicle" in name:
        return 0
    return -1


def save_semantic_map(sample: dict, out_path: Path, frame_idx: int, view_idx: int) -> None:
    clip_index = sample["sam3_clip_index"]
    scene_dir = Path(sample["sam3_scene_dir"])
    t_idx = max(0, min(frame_idx, len(clip_index["frame_ids_10hz"]) - 1))
    v_idx = max(0, min(view_idx, len(clip_index["view_ids"]) - 1))
    view_id = clip_index["view_ids"][v_idx]
    records = _load_scene_records(scene_dir, view_id)
    semantic = np.zeros((DEFAULT_DVGT_OCC_CONFIG.image_height, DEFAULT_DVGT_OCC_CONFIG.image_width, 3), dtype=np.float32)
    alpha = np.zeros((DEFAULT_DVGT_OCC_CONFIG.image_height, DEFAULT_DVGT_OCC_CONFIG.image_width, 1), dtype=np.float32)
    for record_id in clip_index["records_by_frame_view"].get(f"{t_idx}_{v_idx}", []):
        record = records.get(str(record_id))
        if record is None:
            continue
        if (not bool(record.get("quality_ok", False))) or bool(record.get("box_like_flag", False)):
            continue
        cls_idx = _class_to_semantic_idx(str(record.get("class_name", "")))
        if cls_idx < 0:
            continue
        mask = rle_to_binary_mask(record["rle"]).astype(bool)
        if mask.shape != alpha.shape[:2]:
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            mask_img = mask_img.resize((alpha.shape[1], alpha.shape[0]), Image.Resampling.NEAREST)
            mask = np.asarray(mask_img, dtype=np.uint8) > 0
        semantic[mask] = SEMANTIC_PALETTE[cls_idx]
        alpha[mask] = 0.85
    image_path = Path(sample["image_paths"][int(view_id)][t_idx])
    rgb = load_rgb(image_path, (DEFAULT_DVGT_OCC_CONFIG.image_height, DEFAULT_DVGT_OCC_CONFIG.image_width))
    overlay = rgb * (1.0 - alpha) + semantic * alpha
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.imshow(overlay)
    ax.set_title("Semantic Map")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_occ_bev(sample: dict, out_path: Path, frame_idx: int) -> None:
    occ = sample["supervision"]["occ_label"].astype(np.float32)
    t_idx = max(0, min(frame_idx, occ.shape[0] - 1))
    bev = occ[t_idx].max(axis=0)
    extent = [
        DEFAULT_DVGT_OCC_CONFIG.occ_grid.x_range[0],
        DEFAULT_DVGT_OCC_CONFIG.occ_grid.x_range[1],
        DEFAULT_DVGT_OCC_CONFIG.occ_grid.y_range[0],
        DEFAULT_DVGT_OCC_CONFIG.occ_grid.y_range[1],
    ]
    fig, ax = plt.subplots(figsize=(6.5, 6.2), constrained_layout=True)
    ax.imshow(bev, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=max(float(bev.max()), 1e-6))
    ax.set_title("Occupancy BEV (same frame)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample = load_sample(args)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (args.output_root / "figure_assets" / f"scene_{args.scene_id}_{args.clip_id}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    save_rgb_grid(sample, out_dir / "01_multiview_multitime_rgb.png", args.frames, args.views)
    save_pointcloud(sample, out_dir / "02_dvgt_point_scene_ego0.png", args.point_frame, args.max_pointcloud_points)
    save_sam3_masks(
        sample,
        out_dir / "03_sam3_full_mask.png",
        out_dir / "03b_sam3_instance_ids.png",
        frame_idx=args.semantic_frame,
        view_idx=args.semantic_view,
        attention_path=args.sam3_attention_npy,
    )
    save_voxel_volume(sample, out_dir / "04_dense_voxel_volume.png", frame_idx=args.voxel_frame)
    save_semantic_map(sample, out_dir / "05_semantic_map.png", frame_idx=args.semantic_frame, view_idx=args.semantic_view)
    save_occ_bev(sample, out_dir / "06_occ_bev.png", frame_idx=args.voxel_frame)

    print(f"[figure_assets] wrote outputs to {out_dir}")
    for path in sorted(out_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
