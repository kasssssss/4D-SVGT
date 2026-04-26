#!/usr/bin/env bash
set -uo pipefail

ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT"
cd "$ROOT"

unset PYTHONPATH
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/torch_extensions_v64_gsplat"
export MAX_JOBS=1
export MASTER_PORT=29766
export AUTO_RESUME=0
export RESET_OPTIMIZER=1
export LOG_DIR="data/nuscenes_dvgt_v0/logs/train_stage_c_gs_dyn_v65_pergsdyn_sourcefocus_8gpu"
if [[ -f "$LOG_DIR/last.pt" ]]; then
  export RESUME_PATH="$LOG_DIR/last.pt"
else
  export RESUME_PATH="data/nuscenes_dvgt_v0/logs/train_stage_c_gs_dyn_v64_online_dvgt_raydepth_sourcefocus_8gpu/last.pt"
fi

mkdir -p "$LOG_DIR"
bash tools/dvgt_occ/launch_train_ddp.sh \
  configs/dvgt_occ/train_stage_c_gs_dyn_v65_pergsdyn_sourcefocus_8gpu.yaml \
  --log-interval 10 \
  --save-interval 200 \
  --val-interval 400 \
  --val-batches 4 \
  > "$LOG_DIR/train_driver.log" 2>&1
status=$?
echo "$status" > "$LOG_DIR/train_driver.status"
exit "$status"
