"""Training-time qualitative visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dvgt_occ.config import DVGTOccConfig
from dvgt_occ.data.stage_a_utils import DYNAMIC_CLASS_TO_ID


SEMANTIC_CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction",
    "pedestrian",
    "bicycle",
    "motorcycle",
]

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
    ],
    dtype=np.float32,
) / 255.0


def _to_numpy_image(path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((size_hw[1], size_hw[0]), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _resolve_image_path(batch: Mapping[str, object], sample_idx: int, frame_idx: int, view_idx: int) -> Path | None:
    image_paths = batch.get("image_paths")
    if not image_paths:
        return None
    sample_paths = image_paths[sample_idx]
    if not isinstance(sample_paths, Mapping):
        return None
    view_ids = batch.get("view_ids", [])
    sample_view_ids: Sequence[int] = view_ids[sample_idx] if view_ids else list(sample_paths.keys())
    if not sample_view_ids:
        return None
    safe_view_idx = max(0, min(view_idx, len(sample_view_ids) - 1))
    view_id = sample_view_ids[safe_view_idx]
    paths = sample_paths.get(view_id) or sample_paths.get(str(view_id))
    if not paths:
        return None
    safe_frame_idx = max(0, min(frame_idx, len(paths) - 1))
    return Path(paths[safe_frame_idx])


def _subsample_points(xyz: np.ndarray, weights: np.ndarray, max_points: int, quantile: float = 75.0, max_depth: float = 80.0) -> np.ndarray:
    xyz = xyz.reshape(-1, 3)
    weights = weights.reshape(-1)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(weights)
    xyz = xyz[finite]
    weights = weights[finite]
    if xyz.shape[0] == 0:
        return xyz
    keep = weights >= np.percentile(weights, quantile)
    xyz = xyz[keep]
    if xyz.shape[0] == 0:
        return xyz
    depth = np.linalg.norm(xyz, axis=1)
    xyz = xyz[depth <= max_depth]
    if xyz.shape[0] > max_points:
        stride = max(int(np.ceil(xyz.shape[0] / float(max_points))), 1)
        xyz = xyz[::stride]
    return xyz


def _scatter_bev(ax, xyz: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float], title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axhline(0.0, color="white", linewidth=0.5, alpha=0.5)
    ax.axvline(0.0, color="white", linewidth=0.5, alpha=0.5)
    if xyz.shape[0] == 0:
        ax.text(0.5, 0.5, "no points", transform=ax.transAxes, ha="center", va="center")
        return
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], s=1.0, cmap="viridis", alpha=0.7, linewidths=0.0)
    plt.colorbar(scatter, ax=ax, shrink=0.75, fraction=0.046, pad=0.04)


def _semantic_overlay(logits: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, str]:
    logits = logits.astype(np.float32)
    logits = logits - logits.max(axis=0, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=0, keepdims=True).clip(min=1e-6)
    label = probs.argmax(axis=0)
    conf = probs.max(axis=0)
    color = SEMANTIC_PALETTE[label]
    alpha = np.clip((conf - 0.20) / 0.60, 0.0, 1.0)[..., None] * 0.65
    overlay = rgb * (1.0 - alpha) + color * alpha

    present = np.unique(label[conf > 0.35]).tolist()
    legend = ", ".join(SEMANTIC_CLASS_NAMES[idx] for idx in present[:4]) if present else "none"
    if len(present) > 4:
        legend += ", ..."
    return overlay, legend


def save_training_visualization(
    batch: Mapping[str, object],
    outputs: Mapping[str, object],
    config: DVGTOccConfig,
    out_path: Path,
    *,
    step: int,
    epoch: int,
    sample_idx: int = 0,
    frame_idx: int = 0,
    view_idx: int = 0,
    max_scene_points: int = 20000,
) -> None:
    sample_idx = int(max(sample_idx, 0))
    frame_idx = int(max(frame_idx, 0))
    view_idx = int(max(view_idx, 0))

    points = batch["points"][sample_idx, frame_idx].detach().cpu().float().numpy()
    points_conf = batch["points_conf"][sample_idx, frame_idx].detach().cpu().float().numpy()
    gt_scene_xyz = _subsample_points(points, points_conf, max_points=max_scene_points, quantile=75.0)

    gaussians = outputs["gaussians"]
    pred_center = gaussians.center[sample_idx, frame_idx].detach().cpu().float().numpy()
    pred_opacity = gaussians.opacity[sample_idx, frame_idx].detach().cpu().float().numpy().squeeze(-1)
    pred_scene_xyz = _subsample_points(pred_center, pred_opacity, max_points=max_scene_points, quantile=50.0)

    occ_prob = torch.sigmoid(outputs["occ"].occ_logit[sample_idx, frame_idx, 0]).detach().cpu().float().numpy()
    occ_bev = occ_prob.max(axis=0)
    occ_gt = batch["occ_label"][sample_idx, frame_idx].detach().cpu().float().numpy()
    occ_gt_bev = occ_gt.max(axis=0)

    pred_dyn = outputs["render"].render_alpha_dynamic[sample_idx, frame_idx, view_idx, 0].detach().cpu().float().numpy()
    gt_dyn = batch["sam3_dyn_mask_full"][sample_idx, frame_idx, view_idx].detach().cpu().float().numpy()

    sem_logits = outputs["render"].sem_proj_2d[sample_idx, frame_idx, view_idx].detach().cpu().float().numpy()

    image_path = _resolve_image_path(batch, sample_idx=sample_idx, frame_idx=frame_idx, view_idx=view_idx)
    if image_path is not None and image_path.exists():
        rgb = _to_numpy_image(image_path, size_hw=(config.image_height, config.image_width))
    else:
        rgb = np.zeros((config.image_height, config.image_width, 3), dtype=np.float32)

    sem_overlay, sem_legend = _semantic_overlay(sem_logits, rgb)

    x_range = config.occ_grid.x_range
    y_range = config.occ_grid.y_range
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

    pred_rgb = outputs["render"].render_rgb_all[sample_idx, frame_idx, view_idx].detach().cpu().float().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    _scatter_bev(axes[0, 0], gt_scene_xyz, x_range, y_range, "GT Scene Proxy (DVGT metric points)")
    _scatter_bev(axes[0, 1], pred_scene_xyz, x_range, y_range, "Pred Gaussian Scene")

    axes[0, 2].imshow(occ_bev, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 2].contour(occ_gt_bev, levels=[0.5], colors="white", linewidths=0.6, origin="lower", extent=extent)
    axes[0, 2].set_title("Occ Scene (pred heatmap + GT contour)")
    axes[0, 2].set_xlabel("x (m)")
    axes[0, 2].set_ylabel("y (m)")
    axes[0, 2].set_aspect("equal")

    axes[1, 0].imshow(rgb)
    axes[1, 0].set_title("Reference RGB")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(gt_dyn, cmap="Reds", alpha=np.clip(gt_dyn, 0.0, 1.0) * 0.45)
    axes[1, 1].imshow(pred_dyn, cmap="winter", alpha=np.clip(pred_dyn, 0.0, 1.0) * 0.35)
    axes[1, 1].set_title("Dynamic Mask (GT red / Pred cyan)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(pred_rgb.transpose(1, 2, 0).clip(0.0, 1.0))
    axes[1, 2].imshow(sem_overlay, alpha=0.35)
    axes[1, 2].set_title(f"Render RGB + Semantic ({sem_legend})")
    axes[1, 2].axis("off")

    scene_id = batch["scene_id"][sample_idx]
    clip_id = batch["clip_id"][sample_idx]
    frame_id = batch["frame_ids_10hz"][sample_idx][frame_idx]
    view_id = batch["view_ids"][sample_idx][view_idx]
    fig.suptitle(
        f"DVGT-Occ Qualitative | step={step} epoch={epoch} scene={scene_id} clip={clip_id} frame={frame_id} view={view_id}",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
