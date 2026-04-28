#!/usr/bin/env bash
set -uo pipefail

ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT"
cd "$ROOT"

unset PYTHONPATH
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/torch_extensions_v64_gsplat"
export MAX_JOBS=1
export MASTER_PORT=29816
export AUTO_RESUME=0
export LOG_DIR="data/nuscenes_dvgt_v0/logs/train_stage_c_gs_finetune_v88_true288x512_fullres_2gpu"
export RESUME_PATH="data/nuscenes_dvgt_v0/logs/train_stage_c_gs_finetune_v87_fullres_dggtsoft_resethead_init_2gpu/last.pt"
export RESET_OPTIMIZER=1
export RESET_MODULES=0
export RESET_STEP=1

mkdir -p "$LOG_DIR"
bash tools/dvgt_occ/launch_train_ddp.sh \
  configs/dvgt_occ/train_stage_c_gs_finetune_v88_true288x512_fullres_2gpu.yaml \
  --log-interval 10 \
  --save-interval 50 \
  --val-interval 400 \
  --val-batches 2 \
  > "$LOG_DIR/train_driver.log" 2>&1
status=$?
echo "$status" > "$LOG_DIR/train_driver.status"
exit "$status"
