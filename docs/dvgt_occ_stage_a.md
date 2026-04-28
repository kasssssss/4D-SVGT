# DVGT-Occ Stage A Scaffold

This scaffold starts the data and supervision base described in `目标.md`.
DVGT feature/point cache is no longer the main training interface; it is kept
only for debug, speed sanity, and controlled offline ablations.

## Current Scope

- `dvgt_occ.data.manifest` maps `nuscenes_processed_10Hz` scenes into fixed
  2Hz clips with `T=8`, `V=6`, and stable view order.
- `dvgt_occ.data.dataset` can load manifest entries and optional generated
  `.npy` debug cache/supervision artifacts. Main training should run the frozen
  DVGT frontend online so training and inference consume the same RGB interface.
- `dvgt_occ.models` contains importable module shells for token reassembly,
  STH-style dense decoders, Dynamic/Occ/GS heads, streaming query memory, and
  GS-Occ bridges.
- `tools/dvgt_occ` contains the Stage-A command entrypoints.

## First Commands

From the `DVGT` directory inside the `tmux wzy` container:

```bash
python tools/dvgt_occ/prepare_stage_a.py \
  --source-root ../nuscenes_processed_10Hz \
  --output-root data/nuscenes_dvgt_v0 \
  --manifest data/nuscenes_dvgt_v0/manifest_trainval.json
```

Then validate generated inputs:

```bash
python tools/dvgt_occ/validate_manifest.py data/nuscenes_dvgt_v0/manifest_trainval.json \
  --root data/nuscenes_dvgt_v0
```

Use `--strict-outputs` after optional DVGT debug cache, scene-level SAM3 assets, clip-level
SAM3 supervision, track targets, and Occ GT have been generated. The manifest
stores source paths relative to the Stage-A output root, so
`--root data/nuscenes_dvgt_v0` is the expected form.

## Stage-A Generation

Run the first clip end-to-end before launching all clips:

```bash
# Optional debug/offline-ablation cache only; do not use this as the default
# training data interface.
python tools/dvgt_occ/cache_dvgt.py data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --checkpoint pretrained/dvgt2.pt \
  --limit 1

python tools/dvgt_occ/build_track_memory_targets.py data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --limit 1

python tools/dvgt_occ/build_occ_gt.py data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --limit 1
```

SAM3 is now a scene-level primary asset pipeline. Run the long videos first,
then derive clip supervision:

```bash
python tools/dvgt_occ/run_sam3_scene.py data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --sam3-root /path/to/sam3 \
  --sam3-ckpt /path/to/sam3.pt \
  --window-len 64 \
  --overlap 16 \
  --limit-scenes 1

python tools/dvgt_occ/build_sam3_clip_index.py data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --limit 1
```

All heavy preprocessing tools also accept `--shard-index/--num-shards`, and
they honor `RANK/WORLD_SIZE` by default. That makes it easy to launch them with
`torchrun` or manual shard splits across multiple GPUs.

`run_sam3_scene.py` now checkpoints each view after every processed window in
`sam3_scene/view_x/instances.partial.jsonl` plus `progress.json`. If the
cluster reclaims the job, rerun the same command without `--overwrite` and it
will continue from the last completed window.

Finally, smoke-test the generated clip:

```bash
python tools/dvgt_occ/smoke_stage_a.py data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --strict
```

## Next Implementation Hooks

- `tools/dvgt_occ/cache_dvgt.py`: image loading plus `FrozenDVGTWrapper`.
- `tools/dvgt_occ/run_sam3_scene.py`: scene-level prompt loop, overlap stitching,
  quality gate, and weak global id linking.
- `tools/dvgt_occ/build_sam3_clip_index.py`: scene-level asset slicing plus
  `dyn_soft_mask_1_4`, `gs_local_inst_id_1_8`, and quality maps.
- `tools/dvgt_occ/build_track_memory_targets.py`: 10Hz object tracks to 2Hz
  `track_*` arrays.
- `tools/dvgt_occ/build_occ_gt.py`: import `semantic_occ_3d` when available and
  resample it to the ego0 grid, with raw LiDAR fallback.
