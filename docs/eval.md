# Evaluation

We provide a unified evaluation script that runs baselines across multiple benchmarks. It takes a baseline model and evaluation configurations, performs on-the-fly evaluation, and instantly reports the results in a CSV file.

## 1. Benchmarks & Data
You can directly use the `test` [split](../eval/dvgt1/demo_data_geometry.yaml) from the demo dataset to quickly verify the evaluation pipeline, or follow the instructions in [Data Preparation](data_preparation.md) to generate the full test datasets.

---

## 2. Environments & Prerequisites

Depending on the evaluation track, you may need to set up specific environments. 

**For Geometry Evaluation:**
- Native Support: The default environment is fully optimized for the DVGT and VGGT model.
- Extended Models: For MapAnything, CUT3R and StreamVGGT, please refer to their respective requirement files to initialize separate virtual environments.

**For Closed-loop Planning (NavSim):**
- If you wish to evaluate on NavSim v1 and v2, you must install their respective environments. Please note that the environments for the two versions are **not compatible**. Refer to the official installation guide: [navsim](https://github.com/autonomousvision/navsim).
- We assume you have created distinct conda environments named `navsim_v1` and `navsim_v2`.

**For Open-loop Planning (nuScenes):** 
- Refer to the official installation guide: [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit).
- We assume you have created an environment named `nuscenes` for the nuScenes planning evaluation.

---

## 3. Configuration & Evaluation Tracks

Our evaluation covers three main tracks, configured via the `--config` argument in the `eval/` directory. For clarity, we use specific prefixes for different tracks:
* `geometry_*.yaml`: Track 1 - Geometry Evaluation
* `closed_loop_*.yaml`: Track 2 - Closed-loop Planning Evaluation
* `open_loop_*.yaml`: Track 3 - Open-loop Planning Evaluation

> **Note:** The evaluation scripts support multi-node, multi-GPU execution. For simplicity, the following examples demonstrate single-node 8-GPU evaluation.

### Available Evaluation Scripts

**1. DVGT-1**
* **Geometry:**
    * `eval/dvgt1/geometry_demo_data.yaml`
    * `eval/dvgt1/geometry.yaml`

**2. DVGT-2**
* **Geometry:**
    * `eval/dvgt2/geometry_demo_data.yaml`
    * `eval/dvgt2/geometry_default.yaml` *(Stage 2: Fine-tunes streaming and trajectories on all data)*
    * `eval/dvgt2/geometry_pretrain.yaml` *(Stage 1: Pre-trains local point + relative pose + 3D RoPE)*
    * `eval/dvgt2/geometry_default_mlp_head.yaml`
* **Closed-loop:**
    * `eval/dvgt2/closed_loop_test_gt.yaml` *(Tests if ground truth poses are correct)*
    * `eval/dvgt2/closed_loop_default.yaml`
    * `eval/dvgt2/closed_loop_default_finetune_navtrain.yaml` *(Fine-tuned from NavTrain; incorporates ego-vehicle status like velocity, acc, and driving commands as auxiliary input)*
    * `eval/dvgt2/closed_loop_default_mlp_head.yaml`
* **Open-loop:**
    * `eval/dvgt2/open_loop_default.yaml`
    * `eval/dvgt2/open_loop_default_nuscenes.yaml` *(Fine-tuned with ego status)*

**3. Other Models** *(Geometry inference only)*
To run the configurations below, please visit the corresponding repositories to install their specific environments and download checkpoints before execution:
* `eval/cut3r/geometry.yaml` — [CUT3R](https://github.com/CUT3R/CUT3R.git)
* `eval/mapanything/geometry.yaml` — [MapAnything](https://github.com/facebookresearch/map-anything.git)
* `eval/streamvggt/geometry.yaml` — [StreamVGGT](https://github.com/wzzheng/StreamVGGT.git)
* `eval/vggt/geometry.yaml` — *(Note: This can be run directly using the default DVGT environment without additional setup.)*

---

## 4. Run Evaluation

### Track 1: Geometry Evaluation
Evaluates geometric metrics (supported by both DVGT-1 and DVGT-2).

```bash
torchrun \
    --nproc_per_node=8 \
    tools/eval.py \
    --config=eval/dvgt1/geometry.yaml
    # Use --debug to enable verbose logging for easier diagnostics and rapid testing.

# For geometry tests, scenes are evaluated in 16-frame clips. Run this script to merge the 
# clip-level results into scene-level metrics. The final score is averaged across all scenes.
python tools/postprocess/merge_scene_for_geometry_metric.py \
    path/to/your/output/log_dir/metrics_results.csv
```

*Note:* DVGT-2 evaluates using 16-frame direct inference by default. If you prefer to evaluate using streaming inference, you can execute it by specifying the window size:

```bash
torchrun \
    --nproc_per_node=8 \
    tools/eval.py \
    --config=eval/dvgt2/geometry_default.yaml \
    log_dir=output/dvgt2/eval/geometry/official_ckpt_stream_window4 \
    checkpoint_path=ckpt/dvgt2.pt \
    model_wrapper_conf.eval_config.infer_window=4
```

### Track 2: Closed-loop Planning (NavSim v1 / v2)
Evaluates closed-loop planning performance (for DVGT-2). This process involves inference, trajectory generation, and NavSim scoring.

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"

# Prevent `evo` errors during multi-GPU execution
python -c "import evo.tools.settings; print('Settings initialized')"

# Step 1: Model Inference
RESULT_ROOT=output/dvgt2/eval/closed_loop/official_ckpt
RESULT_ROOT=$(realpath -m "$RESULT_ROOT")  # Resolve to absolute path; -m allows for non-existent directories.

torchrun \
    --nproc_per_node=8 \
    tools/eval.py \
    --config=eval/dvgt2/closed_loop_default.yaml \
    log_dir=$RESULT_ROOT

# Step 2: Generate Trajectory JSON
python tools/postprocess/gen_json_traj.py \
    --pth="${RESULT_ROOT}/metrics_results.csv"

# Step 3: Evaluate using NavSim
# ----------------------------------------
# Option A: For NavSim v1
conda activate navsim_v1_1
cd third_party/navsim_v1_1
python run_pdm_score_json.py \
    --pth="${RESULT_ROOT}/trajectory.json"
cd ..

# Option B: For NavSim v2
conda activate navsim_v2_2
cd third_party/navsim_v2_2
python run_pdm_score_json.py \
    --pth="${RESULT_ROOT}/trajectory.json"
cd ..
```

### Track 3: Open-loop Planning (nuScenes)
Evaluates open-loop planning metrics on nuScenes (for DVGT-2). 

Before running, download the nuScenes validation processed `.pkl` file from [OmniDrive](https://github.com/NVlabs/OmniDrive.git) or fetch it directly using the command below, and place it in your nuScenes `data` directory:

```bash
wget https://huggingface.co/datasets/RainyNight/DVGT_demo_dataset/resolve/main/omini_drive_nuscenes2d_ego_temporal_infos_val.pkl
```

```bash
# Step 1: Model Inference
RESULT_ROOT=output/dvgt2/eval/open_loop/official_ckpt
torchrun \
    --nproc_per_node=8 \
    tools/eval.py \
    --config=eval/dvgt2/open_loop_default.yaml \
    log_dir=$RESULT_ROOT

# Step 2: Evaluate Planning Metrics
conda activate nuscenes
python third_party/OmniDrive_nuscene_planning/eval_planning.py \
    --base_path=/path/to/nuscenes/data \
    --pred_path="${RESULT_ROOT}/metrics_results.csv"
```
