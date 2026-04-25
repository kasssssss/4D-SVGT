from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from gsplat.rendering import rasterization


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    out_dir = Path("data/nuscenes_dvgt_v0/logs/dev_gsplat_smoke_20260421")
    out_dir.mkdir(parents=True, exist_ok=True)

    means = torch.tensor(
        [
            [0.0, 0.0, 12.0],
            [-1.5, -0.5, 10.0],
            [1.8, 0.3, 14.0],
        ],
        device=device,
        dtype=dtype,
    )
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9239, 0.0, 0.0, 0.3827],
            [0.9659, 0.0, 0.2588, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    quats = torch.nn.functional.normalize(quats, dim=-1)
    scales = torch.tensor(
        [
            [0.8, 0.4, 0.6],
            [0.5, 1.1, 0.7],
            [1.2, 0.5, 0.5],
        ],
        device=device,
        dtype=dtype,
    )
    opacities = torch.tensor([0.95, 0.85, 0.9], device=device, dtype=dtype)
    colors = torch.tensor(
        [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.4, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    viewmats = torch.eye(4, device=device, dtype=dtype)[None]
    ks = torch.tensor(
        [[[280.0, 0.0, 224.0], [0.0, 280.0, 112.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    )

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=ks,
        width=448,
        height=224,
    )
    rgb = renders[0].detach().float().cpu().numpy()
    alpha = alphas[0, ..., 0].detach().float().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("gsplat rgb")
    plt.subplot(1, 2, 2)
    plt.imshow(alpha, cmap="magma")
    plt.axis("off")
    plt.title("gsplat alpha")
    plt.tight_layout()
    plt.savefig(out_dir / "smoke_gsplat.png", dpi=180)
    plt.close()

    print("rgb_mean", float(renders.mean()))
    print("rgb_max", float(renders.max()))
    print("alpha_mean", float(alphas.mean()))
    print("alpha_max", float(alphas.max()))
    if isinstance(meta, dict):
        print("meta_keys", sorted(meta.keys()))


if __name__ == "__main__":
    main()
