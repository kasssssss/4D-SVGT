"""Training-time qualitative visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dvgt_occ.config import DVGTOccConfig


SEMANTIC_CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction",
    "pedestrian",
    "bicycle",
    "motorcycle",
    "cyclist",
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
        [0, 206, 209],
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


def _scatter_bev_xy(ax, xyz: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float], title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axhline(0.0, color="white", linewidth=0.5, alpha=0.35)
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


def _project_gaussian_debug(
    batch: Mapping[str, object],
    outputs: Mapping[str, object],
    sample_idx: int,
    frame_idx: int,
    view_idx: int,
) -> dict[str, np.ndarray | float | int]:
    gaussians = outputs.get("gaussians")
    if gaussians is None:
        return {"total": 0, "front": 0, "in_frame": 0}

    centers = gaussians.center[sample_idx, frame_idx].detach().cpu().float().numpy().reshape(-1, 3)
    opacity = gaussians.opacity[sample_idx, frame_idx].detach().cpu().float().numpy().reshape(-1)
    scale = gaussians.scale[sample_idx, frame_idx].detach().cpu().float().numpy().reshape(-1, 3).mean(axis=1)
    finite = np.isfinite(centers).all(axis=1) & np.isfinite(opacity) & np.isfinite(scale)
    centers = centers[finite]
    opacity = opacity[finite]
    scale = scale[finite]
    if centers.shape[0] == 0:
        return {"total": 0, "front": 0, "in_frame": 0}

    first_pose = batch["first_ego_pose_world"][sample_idx].detach().cpu().float().numpy()
    camera_to_world = batch["camera_to_world"][sample_idx, frame_idx, view_idx].detach().cpu().float().numpy()
    intr = batch["camera_intrinsics"][sample_idx, view_idx].detach().cpu().float().numpy()
    height = int(batch["rgb_target"].shape[-2])
    width = int(batch["rgb_target"].shape[-1])

    ones = np.ones((centers.shape[0], 1), dtype=np.float32)
    centers_h = np.concatenate([centers.astype(np.float32), ones], axis=1)
    world = (first_pose.astype(np.float32) @ centers_h.T).T[:, :3]
    world_h = np.concatenate([world, ones], axis=1)
    viewmat = np.linalg.inv(camera_to_world.astype(np.float32))
    cam = (viewmat @ world_h.T).T[:, :3]
    z = cam[:, 2]
    front = np.isfinite(z) & (z > 0.5)
    denom = np.where(np.abs(z) > 1e-6, z, 1.0)
    u = intr[0] * cam[:, 0] / denom + intr[2]
    v = intr[1] * cam[:, 1] / denom + intr[3]
    in_frame = front & np.isfinite(u) & np.isfinite(v) & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)

    return {
        "total": int(centers.shape[0]),
        "front": int(front.sum()),
        "in_frame": int(in_frame.sum()),
        "u": u[in_frame],
        "v": v[in_frame],
        "depth": z[in_frame],
        "opacity": opacity[in_frame],
        "scale": scale[in_frame],
        "center_xyz": centers,
        "center_opacity": opacity,
        "center_scale": scale,
        "opacity_mean": float(opacity.mean()),
        "scale_mean": float(scale.mean()),
    }


def _plot_projected_gaussians(ax, rgb: np.ndarray, debug: Mapping[str, object]) -> None:
    ax.imshow(np.clip(rgb * 0.35, 0.0, 1.0))
    u = np.asarray(debug.get("u", np.zeros((0,), dtype=np.float32)))
    v = np.asarray(debug.get("v", np.zeros((0,), dtype=np.float32)))
    if u.size > 0:
        depth = np.asarray(debug.get("depth", np.zeros_like(u)))
        opacity = np.asarray(debug.get("opacity", np.ones_like(u)))
        scale = np.asarray(debug.get("scale", np.ones_like(u) * 0.02))
        if u.size > 30000:
            stride = int(np.ceil(u.size / 30000.0))
            u, v, depth, opacity, scale = u[::stride], v[::stride], depth[::stride], opacity[::stride], scale[::stride]
        sizes = np.clip(scale * 80.0, 0.4, 9.0)
        alphas = np.clip(opacity, 0.15, 0.85)
        scatter = ax.scatter(u, v, c=depth, s=sizes, cmap="turbo", alpha=alphas, linewidths=0.0)
        plt.colorbar(scatter, ax=ax, shrink=0.65, fraction=0.046, pad=0.02)
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.axis("off")
    ax.set_title(
        "Projected Gaussians "
        f"total={int(debug.get('total', 0))} front={int(debug.get('front', 0))} in={int(debug.get('in_frame', 0))}"
    )


def _plot_gaussian_bev(ax, debug: Mapping[str, object], x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
    centers = np.asarray(debug.get("center_xyz", np.zeros((0, 3), dtype=np.float32)))
    opacity = np.asarray(debug.get("center_opacity", np.zeros((0,), dtype=np.float32)))
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if centers.shape[0] > 0:
        if centers.shape[0] > 30000:
            stride = int(np.ceil(centers.shape[0] / 30000.0))
            centers = centers[::stride]
            opacity = opacity[::stride]
        scatter = ax.scatter(centers[:, 0], centers[:, 1], c=opacity, s=1.0, cmap="magma", alpha=0.65, linewidths=0.0)
        plt.colorbar(scatter, ax=ax, shrink=0.65, fraction=0.046, pad=0.02)
    ax.set_title(
        "Gaussian BEV "
        f"opa={float(debug.get('opacity_mean', 0.0)):.3f} scale={float(debug.get('scale_mean', 0.0)):.3f}"
    )


def _target_kind(batch: Mapping[str, object], sample_idx: int, view_idx: int) -> str:
    target_is_source = batch.get("target_is_source")
    if target_is_source is None:
        return "target"
    is_source = bool(target_is_source[sample_idx, view_idx].detach().cpu().item())
    return "source" if is_source else "heldout"


def _choose_source_and_heldout_indices(batch: Mapping[str, object], sample_idx: int, fallback_idx: int) -> tuple[int, int]:
    target_is_source = batch.get("target_is_source")
    view_ids = batch.get("view_ids", [])
    n_views = len(view_ids[sample_idx]) if view_ids else 1
    fallback_idx = max(0, min(int(fallback_idx), max(n_views - 1, 0)))
    if target_is_source is None:
        return fallback_idx, fallback_idx
    flags = target_is_source[sample_idx].detach().cpu().bool().numpy().tolist()
    source_indices = [idx for idx, flag in enumerate(flags) if flag]
    heldout_indices = [idx for idx, flag in enumerate(flags) if not flag]
    source_idx = source_indices[0] if source_indices else fallback_idx
    heldout_idx = heldout_indices[0] if heldout_indices else source_idx
    return source_idx, heldout_idx


def _target_panel(
    batch: Mapping[str, object],
    outputs: Mapping[str, object],
    config: DVGTOccConfig,
    sample_idx: int,
    frame_idx: int,
    view_idx: int,
) -> dict[str, np.ndarray | str | float | bool]:
    image_path = _resolve_image_path(batch, sample_idx=sample_idx, frame_idx=frame_idx, view_idx=view_idx)
    if image_path is not None and image_path.exists():
        rgb = _to_numpy_image(image_path, size_hw=(config.image_height, config.image_width))
    else:
        rgb = np.zeros((config.image_height, config.image_width, 3), dtype=np.float32)

    pred_is_composited = False
    render_out = outputs.get("render")
    if render_out is not None:
        pred_rgb_tensor = render_out.render_rgb_all
        if render_out.render_rgb_all_composited is not None:
            pred_rgb_tensor = render_out.render_rgb_all_composited
            pred_is_composited = True
        if pred_rgb_tensor is not None:
            pred_rgb = pred_rgb_tensor[sample_idx, frame_idx, view_idx].detach().cpu().float().numpy()
        else:
            pred_rgb = np.zeros((3, config.image_height, config.image_width), dtype=np.float32)
        if render_out.render_alpha_all is not None:
            pred_alpha = render_out.render_alpha_all[sample_idx, frame_idx, view_idx, 0].detach().cpu().float().numpy()
        else:
            pred_alpha = np.zeros((config.image_height, config.image_width), dtype=np.float32)
    else:
        pred_rgb = np.zeros((3, config.image_height, config.image_width), dtype=np.float32)
        pred_alpha = np.zeros((config.image_height, config.image_width), dtype=np.float32)

    pred_img = pred_rgb.transpose(1, 2, 0).clip(0.0, 1.0)
    alpha_img = pred_alpha.clip(0.0, 1.0)
    alpha_weight = alpha_img[..., None]
    alpha_tint = np.array([1.0, 0.72, 0.05], dtype=np.float32)
    return {
        "rgb": rgb,
        "pred": pred_img,
        "alpha": alpha_img,
        "overlay": np.clip(0.5 * rgb + 0.5 * pred_img, 0.0, 1.0),
        "abs_err": np.abs(pred_img - rgb).mean(axis=2),
        "signed_resid": np.clip(0.5 + 0.5 * (pred_img - rgb), 0.0, 1.0),
        "pred_on_gt_bg": np.clip(pred_img * alpha_weight + rgb * (1.0 - alpha_weight), 0.0, 1.0),
        "alpha_on_ref": np.clip(rgb * (1.0 - 0.55 * alpha_weight) + alpha_tint * (0.55 * alpha_weight), 0.0, 1.0),
        "pred_exposure": np.clip(pred_img * 3.0, 0.0, 1.0),
        "kind": _target_kind(batch, sample_idx, view_idx),
        "pred_mean": float(pred_img.mean()),
        "pred_max": float(pred_img.max()),
        "alpha_mean": float(alpha_img.mean()),
        "alpha_max": float(alpha_img.max()),
        "pred_is_composited": pred_is_composited,
    }


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

    source_view_idx, heldout_view_idx = _choose_source_and_heldout_indices(batch, sample_idx, view_idx)
    source_panel = _target_panel(batch, outputs, config, sample_idx, frame_idx, source_view_idx)
    heldout_panel = _target_panel(batch, outputs, config, sample_idx, frame_idx, heldout_view_idx)
    audit_panel = heldout_panel if heldout_view_idx != source_view_idx else source_panel
    audit_view_idx = heldout_view_idx if heldout_view_idx != source_view_idx else source_view_idx
    abs_err = np.asarray(audit_panel["abs_err"])
    err_vmax = max(0.08, min(1.0, float(np.percentile(abs_err, 99.0)) if abs_err.size else 0.3))

    fig, axes = plt.subplots(3, 3, figsize=(20, 15), constrained_layout=True)
    axes[0, 0].imshow(source_panel["rgb"])
    axes[0, 0].set_title("Source Reference RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(source_panel["pred"])
    source_pred_title = "Source Pred Composite" if source_panel["pred_is_composited"] else "Source Pred Gaussian"
    axes[0, 1].set_title(
        f"{source_pred_title} mean={source_panel['pred_mean']:.3f} max={source_panel['pred_max']:.3f}"
    )
    axes[0, 1].axis("off")

    source_alpha = np.asarray(source_panel["alpha"])
    alpha_plot = axes[0, 2].imshow(source_alpha, cmap="magma", vmin=0.0, vmax=max(0.2, float(source_alpha.max())))
    axes[0, 2].set_title(f"Source Alpha mean={source_panel['alpha_mean']:.3f} max={source_panel['alpha_max']:.3f}")
    axes[0, 2].axis("off")
    plt.colorbar(alpha_plot, ax=axes[0, 2], shrink=0.65, fraction=0.046, pad=0.02)

    axes[1, 0].imshow(heldout_panel["rgb"])
    axes[1, 0].set_title(f"{heldout_panel['kind'].title()} Reference RGB")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(heldout_panel["pred"])
    heldout_pred_title = f"{heldout_panel['kind'].title()} Pred Composite" if heldout_panel["pred_is_composited"] else f"{heldout_panel['kind'].title()} Pred Gaussian"
    axes[1, 1].set_title(
        f"{heldout_pred_title} mean={heldout_panel['pred_mean']:.3f} max={heldout_panel['pred_max']:.3f}"
    )
    axes[1, 1].axis("off")

    heldout_alpha = np.asarray(heldout_panel["alpha"])
    alpha_plot = axes[1, 2].imshow(heldout_alpha, cmap="magma", vmin=0.0, vmax=max(0.2, float(heldout_alpha.max())))
    axes[1, 2].set_title(
        f"{heldout_panel['kind'].title()} Alpha mean={heldout_panel['alpha_mean']:.3f} max={heldout_panel['alpha_max']:.3f}"
    )
    axes[1, 2].axis("off")
    plt.colorbar(alpha_plot, ax=axes[1, 2], shrink=0.65, fraction=0.046, pad=0.02)

    axes[2, 0].imshow(audit_panel["alpha_on_ref"])
    axes[2, 0].set_title(f"{audit_panel['kind'].title()} Alpha Coverage on Reference")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(audit_panel["pred_on_gt_bg"])
    axes[2, 1].set_title(f"{audit_panel['kind'].title()} Pred on GT Background")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(audit_panel["signed_resid"])
    axes[2, 2].set_title(f"{audit_panel['kind'].title()} Residual |err| mean={abs_err.mean():.3f} max={abs_err.max():.3f}")
    axes[2, 2].axis("off")

    scene_id = batch["scene_id"][sample_idx]
    clip_id = batch["clip_id"][sample_idx]
    frame_id = batch["frame_ids_10hz"][sample_idx][frame_idx]
    view_id = batch["view_ids"][sample_idx][audit_view_idx]
    fig.suptitle(
        f"DVGT-Occ Qualitative | step={step} epoch={epoch} scene={scene_id} clip={clip_id} "
        f"frame={frame_id} audit_view={view_id} source_idx={source_view_idx} heldout_idx={heldout_view_idx}",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
