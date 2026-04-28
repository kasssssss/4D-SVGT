# 4D-SVGT

`4D-SVGT` is my thesis codebase for instance-aware 4D driving scene modeling built on top of DVGT. The project couples:

- online frozen DVGT geometry frontend, with cache kept only for debug/sanity/ablation
- scene-level SAM3 video masks with cross-window stitching
- global weak instance supervision from 3D tracks
- occupancy prediction and Gaussian scene modeling

The current repository is focused on turning that research plan into a trainable engineering stack: data preparation, supervision assets, model scaffolding, training entrypoints, and debugging tools.

## Scope

This repo is not a clean re-release of upstream DVGT. It is a working thesis branch that keeps the original DVGT code where useful and adds a parallel `dvgt_occ` pipeline for:

- Stage-A data preparation plus optional debug caching
- scene-level SAM3 mask generation and clip-level supervision building
- dynamic query / Gaussian / occupancy model wiring
- pilot training, evaluation, and visualization

## Main Directories

- `dvgt_occ/`: thesis model, data, and training code
- `tools/dvgt_occ/`: preprocessing, QC, training, and evaluation scripts
- `configs/dvgt_occ/`: experiment and preprocessing configs
- `docs/`: upstream and project-side documentation that is safe to publish

## Data and Assets

Large datasets, checkpoints, container images, logs, and other machine-local artifacts are intentionally excluded from Git.

Expected runtime assets typically live under:

- `data/nuscenes_dvgt_v0/`
- `pretrained/`

## Training / Preprocessing Entry Points

Common entry scripts live in `tools/dvgt_occ/`, for example:

- `cache_dvgt.py`
- `run_sam3_scene.py`
- `build_sam3_clip_index.py`
- `build_occ_gt.py`
- `train.py`
- `train_overfit.py`

Configs for those flows are under `configs/dvgt_occ/`.

## Upstream Base

This project is built on top of the public DVGT codebase:

- upstream repo: `https://github.com/wzzheng/DVGT`

Upstream code is preserved where needed for geometry reconstruction and reference behavior, while thesis-specific development happens in the added `dvgt_occ` stack.
