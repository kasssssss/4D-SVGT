#!/usr/bin/env python3
"""Smoke-test one or more DVGT-Occ Stage-A clips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ.config import DEFAULT_DVGT_OCC_CONFIG
from dvgt_occ.data.dataset import DVGTOccClipDataset


EXPECTED_SHAPES = {
    "points": (8, 6, 224, 448, 3),
    "points_conf": (8, 6, 224, 448),
    "ego_pose": (8, 7),
    "raw_patch_tokens": (8, 6, 392, 1024),
    "feat_l4": (8, 6, None, 3072),
    "feat_l11": (8, 6, None, 3072),
    "feat_l17": (8, 6, None, 3072),
    "feat_l23": (8, 6, None, 3072),
    "dyn_soft_mask_1_4": (8, 6, 56, 112),
    "gs_local_inst_id_1_8": (8, 6, 28, 56),
    "gs_global_track_id_1_8": (8, 6, 28, 56),
    "inst_boundary_weight_1_8": (8, 6, 28, 56),
    "inst_area_ratio_1_8": (8, 6, 28, 56),
    "inst_quality_mask_1_8": (8, 6, 28, 56),
    "track_boxes_3d": (8, None, 7),
    "track_cls": (8, None),
    "track_id": (8, None),
    "track_visible": (8, None),
    "track_birth": (8, None),
    "track_death": (8, None),
    "track_delta_box": (7, None, 7),
    "occ_label": (8, 10, 100, 100),
    "occ_sem_label": (8, 10, 100, 100),
    "occ_dyn_label": (8, 10, 100, 100),
}

STRICT_CACHE_FILES = [
    "points.npy",
    "points_conf.npy",
    "ego_pose.npy",
    "raw_patch_tokens.npy",
    "feat_l4.npy",
    "feat_l11.npy",
    "feat_l17.npy",
    "feat_l23.npy",
]

STRICT_SUPERVISION_FILES = [
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
]


def _check_shape(name: str, array: np.ndarray) -> None:
    expected = EXPECTED_SHAPES.get(name)
    if expected is None:
        return
    if len(expected) != array.ndim:
        raise AssertionError(f"{name}: expected ndim {len(expected)}, got {array.shape}")
    for dim_idx, dim in enumerate(expected):
        if dim is not None and array.shape[dim_idx] != dim:
            raise AssertionError(f"{name}: expected shape {expected}, got {array.shape}")


def _check_arrays(group: str, arrays: dict, strict: bool) -> None:
    if strict:
        missing = [name for name in EXPECTED_SHAPES if name not in arrays]
        if group == "cache":
            missing = [
                name
                for name in missing
                if name.startswith("points")
                or name.startswith("ego")
                or name.startswith("feat")
                or name == "raw_patch_tokens"
            ]
        elif group == "supervision":
            missing = [
                name
                for name in missing
                if not (
                    name.startswith("points")
                    or name.startswith("ego")
                    or name.startswith("feat")
                    or name == "raw_patch_tokens"
                )
            ]
        if missing:
            raise AssertionError(f"Missing {group} arrays: {missing}")
    for name, array in arrays.items():
        _check_shape(name, array)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise AssertionError(f"{group}/{name} contains NaN or Inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Stage-A generated artifacts.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    dataset = DVGTOccClipDataset(
        manifest_path=args.manifest,
        root=args.output_root or args.manifest.parent,
        load_cache=True,
        load_supervision=True,
        cache_keys=STRICT_CACHE_FILES if args.strict else None,
        supervision_keys=STRICT_SUPERVISION_FILES if args.strict else None,
        projected_semantic_classes=DEFAULT_DVGT_OCC_CONFIG.projected_semantic_classes,
    )
    end = min(len(dataset), args.index + args.limit)
    for idx in range(args.index, end):
        sample = dataset[idx]
        _check_arrays("cache", sample.get("dvgt_cache", {}), strict=args.strict)
        _check_arrays("supervision", sample.get("supervision", {}), strict=args.strict)
        if args.strict and "sam3_clip_index" not in sample:
            raise AssertionError(f"{sample['scene_id']}/{sample['clip_id']} missing sam3_clip_index.json")
        print(
            f"[smoke] {sample['scene_id']}/{sample['clip_id']} "
            f"cache={sorted(sample.get('dvgt_cache', {}).keys())} "
            f"supervision={sorted(sample.get('supervision', {}).keys())}"
        )
    print(f"Smoke checked {end - args.index} clip(s).")


if __name__ == "__main__":
    main()
