#!/usr/bin/env bash
set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT

LOGDIR=data/nuscenes_dvgt_v0/logs/dev_sky_masks_fix_20260420
mkdir -p "$LOGDIR"

export PYTHONPATH=/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT/third_party_shims${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

python3 tools/dvgt_occ/extract_sky_masks_sam3.py \
  data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --sam3-root /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/sam3 \
  --sam3-ckpt /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/sam3_ckpt/sam3.pt \
  --scene-ids 000 \
  --view-ids 0 \
  --overwrite \
  > "$LOGDIR/scene000_view0.log" 2>&1

echo $? > "$LOGDIR/scene000_view0.status"
