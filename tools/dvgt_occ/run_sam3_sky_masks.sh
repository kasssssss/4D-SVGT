#!/usr/bin/env bash
set -euo pipefail

ROOT=/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT
cd "$ROOT"

LOGDIR=data/nuscenes_dvgt_v0/logs/dev_sky_masks_20260420
mkdir -p "$LOGDIR"

export PYTHONPATH=/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT/third_party_shims${PYTHONPATH:+:$PYTHONPATH}

CUDA_VISIBLE_DEVICES=0 python3 tools/dvgt_occ/extract_sky_masks_sam3.py \
  data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --sam3-root /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/sam3 \
  --sam3-ckpt /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/sam3_ckpt/sam3.pt \
  --shard-index 0 \
  --num-shards 2 \
  --overwrite \
  > "$LOGDIR/sky_masks_shard0.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 python3 tools/dvgt_occ/extract_sky_masks_sam3.py \
  data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --sam3-root /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/sam3 \
  --sam3-ckpt /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/sam3_ckpt/sam3.pt \
  --shard-index 1 \
  --num-shards 2 \
  --overwrite \
  > "$LOGDIR/sky_masks_shard1.log" 2>&1 &
pid1=$!

wait $pid0
status0=$?
wait $pid1
status1=$?

echo "$status0" > "$LOGDIR/sky_masks_shard0.status"
echo "$status1" > "$LOGDIR/sky_masks_shard1.status"
if [[ "$status0" -eq 0 && "$status1" -eq 0 ]]; then
  echo 0 > "$LOGDIR/sky_masks_full.status"
else
  echo 1 > "$LOGDIR/sky_masks_full.status"
  exit 1
fi
