from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dvgt_occ.config import DEFAULT_DVGT_OCC_CONFIG
from dvgt_occ.models.rendering.gaussian_mask_renderer import GaussianMaskRenderer
from dvgt_occ.types import GaussianAssignmentOutput, GaussianOutput


def _build_case(case: str, device: torch.device) -> tuple[GaussianOutput, GaussianAssignmentOutput]:
    b = t = v = 1
    h = w = 2
    dense_feat = torch.zeros((b, t, v, h, w, 128), dtype=torch.float32, device=device)
    offset = torch.zeros((b, t, v, h, w, 3), dtype=torch.float32, device=device)
    rotation = torch.zeros((b, t, v, h, w, 4), dtype=torch.float32, device=device)
    rotation[..., 0] = 1.0
    instance_affinity = torch.zeros((b, t, v, h, w, 32), dtype=torch.float32, device=device)
    motion_code = torch.zeros((b, t, v, h, w, 16), dtype=torch.float32, device=device)

    center = torch.zeros((b, t, v, h, w, 3), dtype=torch.float32, device=device)
    scale = torch.ones((b, t, v, h, w, 3), dtype=torch.float32, device=device) * 1.8
    opacity = torch.zeros((b, t, v, h, w, 1), dtype=torch.float32, device=device)
    keep = torch.zeros((b, t, v, h, w, 1), dtype=torch.float32, device=device)
    feat_dc = torch.zeros((b, t, v, h, w, 3), dtype=torch.float32, device=device)

    if case == "a":
        specs = [
            ((0, 0), (-4.0, -1.0, 10.0), (1.0, 0.1, 0.1), 1.6),
            ((0, 1), (4.0, -0.8, 13.0), (0.1, 0.9, 0.2), 2.0),
            ((1, 0), (-1.5, 1.2, 18.0), (0.1, 0.4, 1.0), 2.4),
        ]
    elif case == "b":
        specs = [
            ((0, 0), (-7.0, -1.2, 11.0), (0.9, 0.7, 0.1), 2.2),
            ((0, 1), (0.0, 0.2, 9.0), (0.9, 0.1, 0.9), 1.4),
            ((1, 0), (7.0, -0.5, 14.0), (0.1, 0.9, 0.9), 2.8),
            ((1, 1), (1.5, 2.0, 20.0), (1.0, 1.0, 1.0), 3.2),
        ]
    else:
        raise ValueError(case)

    for (iy, ix), xyz, rgb, s in specs:
        center[0, 0, 0, iy, ix] = torch.tensor(xyz, dtype=torch.float32, device=device)
        feat_dc[0, 0, 0, iy, ix] = torch.tensor(rgb, dtype=torch.float32, device=device)
        scale[0, 0, 0, iy, ix] = s
        opacity[0, 0, 0, iy, ix, 0] = 0.98
        keep[0, 0, 0, iy, ix, 0] = 1.0

    gaussians = GaussianOutput(
        dense_feat=dense_feat,
        center=center,
        offset=offset,
        opacity=opacity,
        scale=scale,
        rotation=rotation,
        feat_dc=feat_dc,
        keep_score=keep,
        instance_affinity=instance_affinity,
        motion_code=motion_code,
        aux_decoder_full=None,
    )
    assignment = GaussianAssignmentOutput(
        assignment_prob=torch.zeros((b, t, v, h, w, 1), dtype=torch.float32, device=device),
        assigned_query=torch.zeros((b, t, v, h, w), dtype=torch.long, device=device),
        background_prob=torch.zeros((b, t, v, h, w, 1), dtype=torch.float32, device=device),
        local_gate=torch.ones((b, t, v, h, w, 1), dtype=torch.float32, device=device),
    )
    return gaussians, assignment


def _export_case(out_dir: Path, case: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DEFAULT_DVGT_OCC_CONFIG
    renderer = GaussianMaskRenderer(output_size=(cfg.image_height, cfg.image_width), splat_radius=cfg.render_splat_radius).to(device)
    gaussians, assignment = _build_case(case, device)
    sem_proj = torch.zeros(
        (1, 1, 1, cfg.projected_semantic_classes, cfg.image_height, cfg.image_width),
        dtype=torch.float32,
        device=device,
    )
    camera_intrinsics = torch.tensor([[[240.0, 240.0, cfg.image_width / 2.0, cfg.image_height / 2.0]]], dtype=torch.float32, device=device)
    camera_to_world = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 1, 4, 4)
    first_ego_pose_world = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4)

    render = renderer(
        gaussians=gaussians,
        assignment=assignment,
        sem_proj_2d=sem_proj,
        camera_intrinsics=camera_intrinsics,
        camera_to_world=camera_to_world,
        first_ego_pose_world=first_ego_pose_world,
        compute_static_branch=True,
        compute_dynamic_rgb=True,
    )

    rgb = render.render_rgb_all[0, 0, 0].detach().cpu().numpy().transpose(1, 2, 0)
    alpha = render.render_alpha_all[0, 0, 0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(np.clip(rgb, 0.0, 1.0))
    axes[0].set_title(f"Synthetic Gaussian Render ({case})")
    axes[0].axis("off")
    axes[1].imshow(alpha, cmap="magma", vmin=0.0, vmax=max(1e-6, float(alpha.max())))
    axes[1].set_title(f"Alpha ({case})")
    axes[1].axis("off")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"synthetic_gaussian_{case}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    out_dir = Path("data/nuscenes_dvgt_v0/logs/dev_synth_gaussian_vis_20260421")
    _export_case(out_dir, "a")
    _export_case(out_dir, "b")
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
