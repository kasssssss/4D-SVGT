#!/usr/bin/env bash
set -euo pipefail

ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT"
cd "$ROOT"

CONFIG="${1:-configs/dvgt_occ/train_stage_d_full.yaml}"
NPROC="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29620}"
LOG_DIR="${LOG_DIR:-data/nuscenes_dvgt_v0/logs/train_stage_d_full}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_PATH="${RESUME_PATH:-}"

shift $(( $# > 0 ? 1 : 0 ))

mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export PYTHONUNBUFFERED=1

if [[ -z "$RESUME_PATH" && "$AUTO_RESUME" == "1" && -f "$LOG_DIR/last.pt" ]]; then
  RESUME_PATH="$LOG_DIR/last.pt"
fi

EXTRA_ARGS=()
if [[ -n "$RESUME_PATH" ]]; then
  EXTRA_ARGS+=(--resume "$RESUME_PATH")
fi

TORCHRUN_BIN="${TORCHRUN_BIN:-}"
if [[ -n "$TORCHRUN_BIN" ]]; then
  TORCHRUN_CMD=("$TORCHRUN_BIN")
elif command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_CMD=("$(command -v torchrun)")
elif command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
  TORCHRUN_CMD=(python -m torch.distributed.run)
elif command -v python3 >/dev/null 2>&1 && python3 -c "import torch" >/dev/null 2>&1; then
  TORCHRUN_CMD=(python3 -m torch.distributed.run)
else
  echo "Could not find a torchrun binary or a Python interpreter with torch installed." >&2
  exit 1
fi

"${TORCHRUN_CMD[@]}" \
  --standalone \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  tools/dvgt_occ/train.py \
  --config "$CONFIG" \
  --log-dir "$LOG_DIR" \
  "${EXTRA_ARGS[@]}" \
  "$@"
