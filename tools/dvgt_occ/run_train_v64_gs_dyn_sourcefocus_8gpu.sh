#!/usr/bin/env bash
set -uo pipefail

ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/DVGT"
cd "$ROOT"

unset PYTHONPATH
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export TORCH_CUDA_ARCH_LIST=8.0
# Keep v64 in a clean gsplat extension cache. Reusing the old shared cache can
# make DDP ranks race on a half-linked gsplat_cuda.so during first rasterization.
export TORCH_EXTENSIONS_DIR="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/driving_data/torch_extensions_v64_gsplat"
export MAX_JOBS=1
export MASTER_PORT=29765
export AUTO_RESUME=0
export RESET_OPTIMIZER=1
export RESUME_PATH="data/nuscenes_dvgt_v0/logs/train_stage_c_gs_only_v63_online_dvgt_raydepth_anchor/last.pt"
export LOG_DIR="data/nuscenes_dvgt_v0/logs/train_stage_c_gs_dyn_v64_online_dvgt_raydepth_sourcefocus_8gpu"

mkdir -p "$LOG_DIR"
bash tools/dvgt_occ/launch_train_ddp.sh \
  configs/dvgt_occ/train_stage_c_gs_dyn_v64_online_dvgt_raydepth_sourcefocus_8gpu.yaml \
  --log-interval 10 \
  --save-interval 200 \
  --val-interval 400 \
  --val-batches 4 \
  > "$LOG_DIR/train_driver.log" 2>&1
status=$?
echo "$status" > "$LOG_DIR/train_driver.status"
exit "$status"
