#!/usr/bin/env bash
set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT

LOGDIR=data/nuscenes_dvgt_v0/logs/dev_backfill_tailfix_20260419
mkdir -p "$LOGDIR"

export PYTHONPATH=/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT/third_party_shims${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0,1

python3 tools/dvgt_occ/run_sam3_scene.py \
  data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --sam3-root /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/sam3 \
  --sam3-ckpt /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/sam3_ckpt/sam3.pt \
  --scene-ids 009 \
  --view-ids 0 \
  --overwrite \
  > "$LOGDIR/scene009_view0.log" 2>&1

echo 0 > "$LOGDIR/scene009_view0.status"
