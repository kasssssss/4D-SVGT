#!/usr/bin/env python3
"""Visualize scene-level SAM3 masks against clip-level supervision."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dvgt_occ import DEFAULT_DVGT_OCC_CONFIG
from dvgt_occ.data import DVGTOccClipDataset
from dvgt_occ.data.stage_a_utils import resize_mask_float, rle_to_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAM3 full-mask vs clip-level supervision consistency.")
    parser.add_argument("--manifest", type=Path, default=Path("data/nuscenes_dvgt_v0/manifest_trainval.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/nuscenes_dvgt_v0"))
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("data/nuscenes_dvgt_v0/logs/dev_sam_refine_20260419/audit_after_scene_refine.json"),
    )
    parser.add_argument("--num-best", type=int, default=2)
    parser.add_argument("--num-worst", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("data/nuscenes_dvgt_v0/qc/sam3_mask_consistency"))
    return parser.parse_args()


def _load_rgb(path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((size_hw[1], size_hw[0]), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _upsample_mask(mask: np.ndarray, size_hw: tuple[int, int], resample: int) -> np.ndarray:
    mask_img = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize((size_hw[1], size_hw[0]), resample)
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def _load_scene_records(scene_dir: Path, view_id: int) -> dict[int, list[dict[str, Any]]]:
    path = scene_dir / f"view_{view_id}" / "instances.jsonl"
    records_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return records_by_frame
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records_by_frame[int(record["frame_id_10hz"])].append(record)
    return records_by_frame


def _choose_frame_view(sample: dict[str, Any]) -> tuple[int, int, float]:
    dyn_soft = sample["supervision"]["dyn_soft_mask_1_4"].astype(np.float32)
    full_dyn = sample["sam3_fullres"]["sam3_dyn_mask_full"].astype(np.float32)
    best = (0, 0, -1.0)
    for t_idx in range(dyn_soft.shape[0]):
        for v_idx in range(dyn_soft.shape[1]):
            full14 = resize_mask_float(full_dyn[t_idx, v_idx], dyn_soft.shape[-2:]).astype(np.float32)
            score = float(np.mean(np.abs(dyn_soft[t_idx, v_idx] - full14)))
            if score > best[2]:
                best = (t_idx, v_idx, score)
    return best


def _scene_mask_panels(records: list[dict[str, Any]], size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    accepted = np.zeros(size_hw, dtype=np.float32)
    rejected = np.zeros(size_hw, dtype=np.float32)
    refined = np.zeros(size_hw, dtype=np.float32)
    stats = {"accepted": 0, "rejected": 0, "refined": 0, "box_like": 0}
    for record in records:
        mask = rle_to_binary_mask(record["rle"]).astype(np.float32)
        if mask.shape != size_hw:
            mask = resize_mask_float(mask, size_hw).astype(np.float32)
        if bool(record.get("quality_ok", False)) and not bool(record.get("box_like_flag", False)):
            accepted = np.maximum(accepted, mask)
            stats["accepted"] += 1
        else:
            rejected = np.maximum(rejected, mask)
            stats["rejected"] += 1
        if bool(record.get("refined", False)):
            refined = np.maximum(refined, mask)
            stats["refined"] += 1
        if bool(record.get("box_like_flag", False)):
            stats["box_like"] += 1
    return np.stack([accepted, rejected, refined], axis=-1), accepted, stats


def _render_case(sample: dict[str, Any], audit_case: dict[str, Any], out_path: Path) -> None:
    image_hw = (DEFAULT_DVGT_OCC_CONFIG.image_height, DEFAULT_DVGT_OCC_CONFIG.image_width)
    t_idx, v_idx, mae = _choose_frame_view(sample)
    view_id = int(sample["view_ids"][v_idx])
    frame_id = int(sample["frame_ids_10hz"][t_idx])
    image_path = Path(sample["image_paths"][view_id][t_idx])
    rgb = _load_rgb(image_path, image_hw)

    scene_dir = Path(sample["sam3_scene_dir"])
    records = _load_scene_records(scene_dir, view_id).get(frame_id, [])
    quality_rgb, accepted_mask, stats = _scene_mask_panels(records, image_hw)

    full_dyn = sample["sam3_fullres"]["sam3_dyn_mask_full"][t_idx, v_idx].astype(np.float32)
    dyn14 = sample["supervision"]["dyn_soft_mask_1_4"][t_idx, v_idx].astype(np.float32)
    dyn14_up = _upsample_mask(dyn14, image_hw, Image.Resampling.BILINEAR)
    full14 = resize_mask_float(full_dyn, dyn14.shape).astype(np.float32)
    full14_up = _upsample_mask(full14, image_hw, Image.Resampling.NEAREST)

    bin_dyn14 = dyn14 > 0.5
    bin_full14 = full14 > 0.5
    fp = np.logical_and(bin_dyn14, ~bin_full14).astype(np.float32)
    fn = np.logical_and(bin_full14, ~bin_dyn14).astype(np.float32)
    tp = np.logical_and(bin_dyn14, bin_full14).astype(np.float32)
    diff_rgb = np.stack([fn, tp, fp], axis=-1)
    diff_rgb = _upsample_mask(diff_rgb[..., 0], image_hw, Image.Resampling.NEAREST)[..., None] * np.array([1.0, 0.0, 0.0])[None, None, :] + \
        _upsample_mask(diff_rgb[..., 1], image_hw, Image.Resampling.NEAREST)[..., None] * np.array([0.0, 1.0, 0.0])[None, None, :] + \
        _upsample_mask(diff_rgb[..., 2], image_hw, Image.Resampling.NEAREST)[..., None] * np.array([0.0, 0.5, 1.0])[None, None, :]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB\nscene={sample['scene_id']} clip={sample['clip_id']} t={t_idx} view={view_id}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb)
    axes[0, 1].imshow(full_dyn, cmap="Reds", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("scene-level full dyn mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(rgb)
    axes[0, 2].imshow(quality_rgb, alpha=0.45)
    axes[0, 2].set_title(
        f"quality overlay\nacc={stats['accepted']} rej={stats['rejected']} ref={stats['refined']} box={stats['box_like']}"
    )
    axes[0, 2].axis("off")

    axes[1, 0].imshow(rgb)
    axes[1, 0].imshow(dyn14_up, cmap="Blues", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("clip dyn_soft_mask_1_4 (upsampled)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(diff_rgb)
    axes[1, 1].set_title(
        f"downsample compare\nred=full-only green=agree cyan=clip-only\nIoU={audit_case['dyn14_vs_full14_iou']:.3f} MAE={mae:.4f}"
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(rgb)
    axes[1, 2].imshow(accepted_mask, cmap="Greens", alpha=0.35, vmin=0.0, vmax=1.0)
    axes[1, 2].imshow(np.clip(quality_rgb[..., 1], 0.0, 1.0), cmap="Reds", alpha=0.30, vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("accepted full mask + rejected highlights")
    axes[1, 2].axis("off")

    fig.suptitle(
        f"SAM3 QC | scene={sample['scene_id']} clip={sample['clip_id']} | audit_iou={audit_case['dyn14_vs_full14_iou']:.3f}",
        fontsize=16,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    sample_cases = list(audit["sample_clips"])
    sample_cases.sort(key=lambda item: float(item["dyn14_vs_full14_iou"]))
    selected = sample_cases[: args.num_worst] + sample_cases[-args.num_best :]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for case in selected:
        dataset = DVGTOccClipDataset(
            manifest_path=args.manifest,
            root=args.output_root.resolve(),
            load_cache=False,
            load_supervision=True,
            load_scene_sam3_full=True,
            scene_ids=[case["scene_id"]],
            clip_ids=[case["clip_id"]],
            limit=1,
            full_res_size=(DEFAULT_DVGT_OCC_CONFIG.image_height, DEFAULT_DVGT_OCC_CONFIG.image_width),
            projected_semantic_classes=DEFAULT_DVGT_OCC_CONFIG.projected_semantic_classes,
        )
        if len(dataset) == 0:
            continue
        sample = dataset[0]
        out_path = out_dir / f"sam3_qc_scene_{case['scene_id']}_{case['clip_id']}.png"
        _render_case(sample, case, out_path)
        summary_rows.append(
            {
                "scene_id": case["scene_id"],
                "clip_id": case["clip_id"],
                "dyn14_vs_full14_iou": float(case["dyn14_vs_full14_iou"]),
                "path": str(out_path),
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "num_visualizations": len(summary_rows)}, ensure_ascii=False))
    for row in summary_rows:
        print(f"{row['scene_id']}/{row['clip_id']} iou={row['dyn14_vs_full14_iou']:.4f} -> {row['path']}")


if __name__ == "__main__":
    main()
