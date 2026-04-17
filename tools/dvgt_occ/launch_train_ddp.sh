#!/usr/bin/env bash
set -euo pipefail

ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT"
cd "$ROOT"

CONFIG="${1:-configs/dvgt_occ/train_stage_d_full.yaml}"
NPROC="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29620}"
LOG_DIR="${LOG_DIR:-data/nuscenes_dvgt_v0/logs/train_stage_d_full}"

shift $(( $# > 0 ? 1 : 0 ))

mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export PYTHONUNBUFFERED=1

torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  tools/dvgt_occ/train.py \
  --config "$CONFIG" \
  --log-dir "$LOG_DIR" \
  "$@"
