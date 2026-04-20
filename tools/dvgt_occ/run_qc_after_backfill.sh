#!/usr/bin/env bash
set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT

LOGDIR=data/nuscenes_dvgt_v0/logs/dev_backfill_full_20260419
mkdir -p "$LOGDIR"

python3 tools/dvgt_occ/qc_sam3_mask_consistency.py \
  --manifest data/nuscenes_dvgt_v0/manifest_trainval.json \
  --output-root data/nuscenes_dvgt_v0 \
  --audit-json "$LOGDIR/audit_after_backfill.json" \
  --num-worst 2 \
  --num-best 2 \
  --out-dir data/nuscenes_dvgt_v0/qc/sam3_mask_consistency_backfill_full \
  > "$LOGDIR/qc_after_backfill.log" 2>&1

echo 0 > "$LOGDIR/qc_after_backfill.status"
echo 0 > "$LOGDIR/postprocess_after_backfill.status"
