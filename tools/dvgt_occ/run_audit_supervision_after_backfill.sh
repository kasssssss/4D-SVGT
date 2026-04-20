#!/usr/bin/env bash
set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT

LOGDIR=data/nuscenes_dvgt_v0/logs/dev_backfill_full_20260419
mkdir -p "$LOGDIR"

python3 tools/dvgt_occ/audit_supervision_assets.py \
  --manifest data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --sample-clips 6 \
  --out "$LOGDIR/audit_after_backfill.json" \
  > "$LOGDIR/audit_after_backfill.log" 2>&1

echo 0 > "$LOGDIR/audit_after_backfill.status"
