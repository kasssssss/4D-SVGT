#!/usr/bin/env python3
"""Import and resample scene occupancy assets to the DVGT-Occ ego0 grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dvgt_occ.config import DEFAULT_DVGT_OCC_CONFIG
from dvgt_occ.data.manifest import load_manifest
from dvgt_occ.data.stage_a_utils import (
    build_track_arrays,
    distributed_shard_spec,
    entry_output_dirs,
    filter_manifest_entries,
    load_occ_asset,
    load_lidar_points,
    load_lidar_pose,
    required_files_exist,
    shard_items,
    transform_points,
    voxelize_points,
)


OUTPUT_FILES = ("occ_label.npy", "occ_sem_label.npy", "occ_dyn_label.npy")


def _semantic_occ_to_target(scene_dir: Path, frame_id: int, first_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    asset_path = scene_dir / "semantic_occ_3d" / f"{frame_id:03d}.npz"
    if not asset_path.exists():
        return None
    asset = load_occ_asset(asset_path)
    occupied = asset["occupied_mask"].astype(bool)
    if not occupied.any():
        grid = DEFAULT_DVGT_OCC_CONFIG.occ_grid
        shape = grid.shape_zyx
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)

    source_idx = np.argwhere(occupied)
    voxel_size = asset["voxel_size"].astype(np.float32)
    xyz_min = asset["xyz_min"].astype(np.float32)
    coords = xyz_min[None, :] + (source_idx.astype(np.float32) + 0.5) * voxel_size[None, :]
    coords_first = transform_points(coords, first_inv @ asset["lidar_to_world"])

    grid = DEFAULT_DVGT_OCC_CONFIG.occ_grid
    nz, ny, nx = grid.shape_zyx
    occ = np.zeros((nz, ny, nx), dtype=np.uint8)
    sem = np.zeros((nz, ny, nx), dtype=np.uint8)
    mins = np.asarray([grid.x_range[0], grid.y_range[0], grid.z_range[0]], dtype=np.float32)
    ijk = np.floor((coords_first - mins) / grid.voxel_size).astype(np.int64)
    ix, iy, iz = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    source_sem = asset["semantics"][occupied].astype(np.uint8)
    for src_sem, x, y, z in zip(source_sem[valid], ix[valid], iy[valid], iz[valid]):
        occ[z, y, x] = 1
        remapped = 0 if int(src_sem) == 255 else min(int(src_sem), 17)
        sem[z, y, x] = max(sem[z, y, x], remapped)
    return occ, sem


def _mark_dynamic_voxels(track_arrays: dict, occ_label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = DEFAULT_DVGT_OCC_CONFIG.occ_grid
    nz, ny, nx = grid.shape_zyx
    xs = np.arange(nx, dtype=np.float32) * grid.voxel_size + grid.x_range[0] + grid.voxel_size * 0.5
    ys = np.arange(ny, dtype=np.float32) * grid.voxel_size + grid.y_range[0] + grid.voxel_size * 0.5
    zs = np.arange(nz, dtype=np.float32) * grid.voxel_size + grid.z_range[0] + grid.voxel_size * 0.5
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    centers = np.stack([xx, yy, zz], axis=-1)

    dyn = np.zeros_like(occ_label, dtype=np.uint8)
    sem = np.where(occ_label > 0, 1, 0).astype(np.uint8)
    boxes = track_arrays["track_boxes_3d"]
    visible = track_arrays["track_visible"]
    cls = track_arrays["track_cls"]
    for t_idx in range(occ_label.shape[0]):
        occupied = occ_label[t_idx] > 0
        for obj_idx in np.flatnonzero(visible[t_idx]):
            box = boxes[t_idx, obj_idx]
            cx, cy, cz, dx, dy, dz, yaw = box
            rel = centers - np.array([cx, cy, cz], dtype=np.float32)
            cos_y = np.cos(-yaw)
            sin_y = np.sin(-yaw)
            local_x = rel[..., 0] * cos_y - rel[..., 1] * sin_y
            local_y = rel[..., 0] * sin_y + rel[..., 1] * cos_y
            inside = (
                (np.abs(local_x) <= dx * 0.5)
                & (np.abs(local_y) <= dy * 0.5)
                & (np.abs(rel[..., 2]) <= dz * 0.5)
                & occupied
            )
            dyn[t_idx, inside] = 1
            if cls[t_idx, obj_idx] >= 0:
                sem[t_idx, inside] = np.uint8(cls[t_idx, obj_idx] + 2)
    return sem, dyn


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SurroundOcc-style occupancy labels.")
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
    built = 0
    skipped = 0
    for entry in entries:
        dirs = entry_output_dirs(entry, output_root)
        sup_dir = dirs["supervision"]
        sup_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite and required_files_exist(sup_dir, OUTPUT_FILES):
            skipped += 1
            continue

        frame_ids = entry["frame_ids_10hz"]
        scene_dir = dirs["source_scene"]
        first_inv = np.linalg.inv(load_lidar_pose(scene_dir, frame_ids[0]))
        occ_frames = []
        occ_sem_frames = []
        for frame_id in frame_ids:
            imported = _semantic_occ_to_target(scene_dir, frame_id, first_inv)
            if imported is not None:
                occ_frame, sem_frame = imported
            else:
                points = load_lidar_points(scene_dir / "lidar" / f"{frame_id:03d}.bin")
                lidar_to_first = first_inv @ load_lidar_pose(scene_dir, frame_id)
                points_first = transform_points(points, lidar_to_first)
                occ_frame = voxelize_points(points_first, DEFAULT_DVGT_OCC_CONFIG.occ_grid)
                sem_frame = np.where(occ_frame > 0, 1, 0).astype(np.uint8)
            occ_frames.append(occ_frame)
            occ_sem_frames.append(sem_frame)
        occ_label = np.stack(occ_frames, axis=0).astype(np.uint8)
        occ_sem_label = np.stack(occ_sem_frames, axis=0).astype(np.uint8)
        track_arrays = build_track_arrays(scene_dir, frame_ids)
        dyn_sem_label, occ_dyn_label = _mark_dynamic_voxels(track_arrays, occ_label)
        occ_sem_label = np.maximum(occ_sem_label, dyn_sem_label)

        np.save(sup_dir / "occ_label.npy", occ_label)
        np.save(sup_dir / "occ_sem_label.npy", occ_sem_label)
        np.save(sup_dir / "occ_dyn_label.npy", occ_dyn_label)
        built += 1
        print(
            f"[occ][{shard_index}/{num_shards}] {entry['scene_id']}/{entry['clip_id']} "
            f"occupied={int(occ_label.sum())} dynamic={int(occ_dyn_label.sum())}"
        )
    print(f"Built occupancy GT for {built} clips; skipped {skipped}; shard {shard_index}/{num_shards}.")


if __name__ == "__main__":
    main()
