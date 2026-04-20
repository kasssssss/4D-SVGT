#!/usr/bin/env bash
set -euo pipefail

ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT"
LOGDIR="$ROOT/data/nuscenes_dvgt_v0/logs/dev_scenelevel_qc_20260419_v2"
OUTDIR="$ROOT/data/nuscenes_dvgt_v0/qc/sam3_scene_assets_20260419_v2"

mkdir -p "$LOGDIR" "$OUTDIR"

cd "$ROOT"
python3 -m py_compile tools/dvgt_occ/qc_sam3_scene_assets.py > "$LOGDIR/py_compile.log" 2>&1
echo 0 > "$LOGDIR/py_compile.status"

python3 tools/dvgt_occ/qc_sam3_scene_assets.py \
  --manifest data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --out-dir data/nuscenes_dvgt_v0/qc/sam3_scene_assets_20260419_v2 \
  > "$LOGDIR/run.log" 2>&1
echo 0 > "$LOGDIR/run.status"
