#!/usr/bin/env bash
set -euo pipefail

ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT"
EXT_ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/torch_extensions_80"
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
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DEVICE_ARCH="$(
  python3 - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
PY
)"

if [[ -n "$DEVICE_ARCH" && -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST="$DEVICE_ARCH"
fi

if [[ -n "$DEVICE_ARCH" && -z "${TORCH_EXTENSIONS_DIR:-}" ]]; then
  export TORCH_EXTENSIONS_DIR="$EXT_ROOT"
fi

mkdir -p "${TORCH_EXTENSIONS_DIR:-$EXT_ROOT}"
export MAX_JOBS="${MAX_JOBS:-1}"

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

GSP_LAT_SO="${TORCH_EXTENSIONS_DIR}/gsplat_cuda/gsplat_cuda.so"
GSP_LAT_LOCK="${TORCH_EXTENSIONS_DIR}/gsplat_cuda/lock"

if [[ -f "$GSP_LAT_SO" ]]; then
  rm -f "$GSP_LAT_LOCK"
  echo "[launch_train_ddp] reusing existing gsplat extension at $GSP_LAT_SO" >&2
else
  echo "[launch_train_ddp] prewarming gsplat extension into ${TORCH_EXTENSIONS_DIR:-<default>}" >&2
  PYTHONPATH=. python3 - <<'PY' >/dev/null
import torch
from gsplat.rendering import rasterization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
means = torch.tensor([[0.0, 0.0, 12.0]], device=device, dtype=torch.float32)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
scales = torch.tensor([[0.8, 0.4, 0.6]], device=device, dtype=torch.float32)
opacities = torch.tensor([0.95], device=device, dtype=torch.float32)
colors = torch.tensor([[1.0, 0.2, 0.2]], device=device, dtype=torch.float32)
viewmats = torch.eye(4, device=device, dtype=torch.float32)[None]
Ks = torch.tensor(
    [[[280.0, 0.0, 224.0], [0.0, 280.0, 112.0], [0.0, 0.0, 1.0]]],
    device=device,
    dtype=torch.float32,
)
rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=448,
    height=224,
)
PY
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
