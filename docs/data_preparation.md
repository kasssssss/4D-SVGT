# Data Preparation

This document provides instructions for preparing the training data for our model. We provide scripts to process various datasets (nuScenes, DDAD, KITTI, Waymo, nuPlan/OpenScene/NavSim, Argoverse 2) into a unified Parquet format required for training. This pipeline is highly extensible for adding new datasets. 

By default, our scripts process data at **2Hz**, but **10Hz** generation is also supported for datasets that provide it.

## Table of Contents
- [Quick Start: Demo Dataset](#quick-start-demo-dataset)
- [Preparation](#preparation)
  - [1. Download Raw Datasets](#1-download-raw-datasets)
  - [2. Download Pre-trained Weights](#2-download-pre-trained-weights)
  - [3. Main Environment Setup](#3-main-environment-setup)
- [Dataset Processing Pipeline](#dataset-processing-pipeline)
  - [nuScenes](#nuscenes)
  - [KITTI](#kitti)
  - [Waymo](#waymo)
  - [DDAD](#ddad)
  - [Argoverse 2](#argoverse-2)
  - [OpenScene](#openscene)
  - [NavSim](#navsim)
  - [nuPlan (10Hz)](#nuplan-10hz)
- [DVGT2: Anchor Generation](#dvgt2-anchor-generation)
- [Final Directory Structure](#final-directory-structure)

---

## Quick Start: Demo Dataset

If you want to quickly run through our training, inference, or fine-tuning pipelines without processing massive datasets, you can download our demo dataset directly from Hugging Face (only 551 MB):

```bash
wget https://huggingface.co/datasets/RainyNight/DVGT_demo_dataset/resolve/main/dvgt_demo_data.tar.gz
tar -xzvf dvgt_demo_data.tar.gz
````

-----

## Preparation

### 1\. Download Raw Datasets

Please refer to the official documentation of each dataset to download the raw files:

  - [nuScenes](https://github.com/nutonomy/nuscenes-devkit)
  - [DDAD](https://github.com/TRI-ML/DDAD.git)
  - [Waymo](https://github.com/waymo-research/waymo-open-dataset): We follow [Street-Gaussians-ns](https://github.com/LightwheelAI/street-gaussians-ns.git) for Waymo processing and use Waymo perception\_v1.4.0 for testing.
  - [KITTI](https://www.cvlibs.net/datasets/kitti/): You can use this [script](https://github.com/pean1128/KITTI-download-scipt/blob/master/raw_data_downloader) to quickly download all necessary files.
  - [Argoverse 2](https://github.com/argoverse/av2-api)
  - [nuPlan](https://github.com/motional/nuplan-devkit):
      - For **10Hz** training data, download the full nuPlan dataset.
      - For **2Hz** training data, download [OpenScene](https://github.com/OpenDriveLab/OpenScene).
      - If you only need `navtrain`, download [NavSim](https://github.com/autonomousvision/navsim.git).

Place all downloaded datasets into a centralized `public_datasets` directory:

```text
public_datasets
├── nuscenes
├── ddad
├── waymo
├── kitti
├── nuplan-v1.1
├── openscene
│   └── navsim
└── argoverse2
```

### 2\. Download Pre-trained Weights

Our pipeline relies on [MoGe v2](https://github.com/microsoft/moge) for annotation. Please download the [MoGe v2 weights](https://huggingface.co/Ruicheng/moge-2-vitl) and place them in `ckpt/moge-2-vitl`.

For DVGT1, we use the [VGGT-1B pre-trained weights](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt). Download and place them in `ckpt/VGGT-1B`.

### 3\. Main Environment Setup

We provide a unified environment for nuScenes, DDAD, KITTI, and OpenScene. *(Note: Waymo, Argoverse 2, and nuPlan 10Hz require specific environments, which are detailed in their respective sections below).*

All environments have been tested on **Ubuntu 22.04 with CUDA 11.8.0**.

```bash
# Create and activate conda environment
conda create -n dvgt_data python=3.10 -y
conda activate dvgt_data
pip install --upgrade pip

# Core dependencies
pip install numpy==1.26.4
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# MoGe dependencies
pip install -r third_party/MoGe/requirements.txt

# nuScenes
pip install nuscenes-devkit 

# DDAD
cd third_party
git clone https://github.com/TRI-ML/dgp.git
cd dgp
pip install -e .
cd ../..

# KITTI
pip install pykitti
```

-----

## Dataset Processing Pipeline

All generation scripts support multi-node, multi-GPU processing using standard `torchrun`. However, **we highly recommend using 2 nodes with 8 GPUs each for annotation**, as the scripts gather all data to `rank0` for disk writing; too many nodes may cause an I/O bottleneck. For simplicity, the examples below demonstrate single-node 8-GPU execution. The filtering scripts mainly read existing images to compute metrics, so a single 8-GPU node is sufficient.

### nuScenes

**Prerequisite:** You need the `pkl` file from [OccWorld](https://github.com/wzzheng/OccWorld) to extract drive command information. Alternatively, you can run the following commands to download and extract it directly:

```bash
wget https://huggingface.co/datasets/RainyNight/DVGT_demo_dataset/resolve/main/occworld_nuscenes_pkl.zip
unzip occworld_nuscenes_pkl.zip -d public_datasets/nuscene_occ_world_pkl
```

**Step 1: Generate Depth Data**

```bash
conda activate dvgt_data

# Process validation split
torchrun --nproc_per_node=8 tools/data_process/main.py nuscene \
    --split=val --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image

# Process training split
torchrun --nproc_per_node=8 tools/data_process/main.py nuscene \
    --split=train --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image
```

**Step 2: Filter Data**

```bash
# Evaluate depth alignment
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py nuscenes --split=val --eval_align_depth
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py nuscenes --split=train --eval_align_depth

# Optional: Visualize training metrics to guide filtering
python tools/data_process/filter_data/visualize.py nuscenes --split=train

# Filter roughly 15%-20% of the data based on configured metrics
python tools/data_process/filter_data/filter_data.py nuscene --split=val
python tools/data_process/filter_data/filter_data.py nuscene --split=train
```

-----

### KITTI

**Step 1: Generate Depth Data**

```bash
conda activate dvgt_data

# Process validation split
torchrun --nproc_per_node=8 tools/data_process/main.py kitti \
    --split=val --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image

# Process training split
torchrun --nproc_per_node=8 tools/data_process/main.py kitti \
    --split=train --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image
```

**Step 2: Filter Data**

```bash
# Evaluate depth alignment
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py kitti --split=val --eval_align_depth
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py kitti --split=train --eval_align_depth

# Optional: Visualize training metrics to guide filtering
python tools/data_process/filter_data/visualize.py kitti --split=train

# Filter roughly 15%-20% of the data based on configured metrics
python tools/data_process/filter_data/filter_data.py kitti --split=val
python tools/data_process/filter_data/filter_data.py kitti --split=train
```

-----

### Waymo

Waymo requires TensorFlow, so it uses a dedicated environment and preprocessing step.

**Step 1: Environment & Preprocessing**

```bash
conda create -n waymo python=3.10 -y
conda activate waymo
pip install --upgrade pip numpy scipy tqdm open3d
pip install tensorflow==2.11.0 waymo-open-dataset-tf-2-11-0

# Preprocess raw data
python tools/data_process/preprocess/waymo.py \
    -i public_datasets/waymo \
    -o public_datasets/waymo_ns
```

**Step 2: Generate Depth Data**

```bash
conda activate dvgt_data

# Process validation split
torchrun --nproc_per_node=8 tools/data_process/main.py waymo \
    --split=val --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image

# Process training split
torchrun --nproc_per_node=8 tools/data_process/main.py waymo \
    --split=train --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image
```

**Step 3: Filter Data**

```bash
# Evaluate depth alignment
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py waymo --split=val --eval_align_depth
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py waymo --split=train --eval_align_depth

# Optional: Visualize training metrics to guide filtering
python tools/data_process/filter_data/visualize.py waymo --split=train

# Filter roughly 15%-20% of the data based on configured metrics
python tools/data_process/filter_data/filter_data.py waymo --split=val
python tools/data_process/filter_data/filter_data.py waymo --split=train
```

-----

### DDAD

**Step 1: Generate Depth Data**

```bash
conda activate dvgt_data

# Process validation split
torchrun --nproc_per_node=8 tools/data_process/main.py ddad \
    --split=val --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image

# Process training split
torchrun --nproc_per_node=8 tools/data_process/main.py ddad \
    --split=train --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image
```

**Step 2: Filter Data**

```bash
# Evaluate depth alignment
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py ddad --split=val --eval_align_depth
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py ddad --split=train --eval_align_depth

# Optional: Visualize training metrics to guide filtering
python tools/data_process/filter_data/visualize.py ddad --split=train

# Filter roughly 15%-20% of the data based on configured metrics
python tools/data_process/filter_data/filter_data.py ddad --split=val
python tools/data_process/filter_data/filter_data.py ddad --split=train
```

-----

### Argoverse 2

Argoverse 2 requires a dedicated environment for its initial preprocessing.

**Step 1: Environment & Preprocessing**

```bash
conda create -n av2 python=3.10 -y
conda activate av2
pip install --upgrade pip numpy pandas scipy tqdm opencv-python pyquaternion pyarrow av2

# Preprocess raw data
python process_av2.py \
  --input_dir public_datasets/argoverse2/sensor \
  --output_dir public_datasets/argoverse2_process
```

**Step 2: Generate Depth Data**

```bash
conda activate dvgt_data

# Process test split
torchrun --nproc_per_node=8 tools/data_process/main.py argoverse \
    --split=test --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image

# Process training split
torchrun --nproc_per_node=8 tools/data_process/main.py argoverse \
    --split=train --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image
```

-----

### OpenScene

OpenScene is extremely large. We recommend splitting the training set into 5 slices (`trainval_slice_i`, where `i=[1, 2, 3, 4, 5]`) and annotating each slice using 2 nodes with 8 GPUs.

**Step 1: Generate Depth Data**

```bash
conda activate dvgt_data

# Process test split
torchrun --nproc_per_node=8 tools/data_process/main.py openscene \
    --split=test --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image

# Process training split slices (run for i=1 to 5)
torchrun --nproc_per_node=8 tools/data_process/main.py openscene \
    --split=trainval_slice_i \
    --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image
```

**Step 2: Filter Data**

```bash
# Evaluate depth alignment for test split
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py openscene --split=test --eval_align_depth

# Evaluate depth alignment for train slices (run for i=1 to 5)
torchrun --nproc_per_node=8 tools/data_process/filter_data/evaluate.py openscene \
    --split=trainval_slice_i --eval_align_depth

# Optional: Visualize training metrics to guide filtering
python tools/data_process/filter_data/visualize.py openscene --split=trainval_slice_i

# Filter data
python tools/data_process/filter_data/filter_data.py openscene --split=test
python tools/data_process/filter_data/filter_data.py openscene --split=trainval_slice_i 

# Finally, concatenate the filtered slices into a single training dataset
python tools/data_process/postprocess/openscene_concat_train_data.py

# Optional: Evaluating on the entire OpenScene test set during training can be quite slow. We recommend downsampling the test data to 10% for faster validation during the training process:
python tools/data_process/postprocess/downsample_parquet_data.py
```

-----

### NavSim

If you have already generated OpenScene data, you only need to generate the metadata here, as OpenScene includes NavSim. If you haven't processed OpenScene, append `--gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image` to the commands below.

*Note: We do not train the primary model using this dataset because our model requires a variable context window, whereas NavSim clips are fixed at 14 frames. However, we use this data to fine-tune and achieve better performance on the NavSim v1/v2 planning leaderboards.*

```bash
conda activate dvgt_data

# Process test split
torchrun --nproc_per_node=8 tools/data_process/main.py navsim \
    --split=test --gen_meta

# Process training slices (run for i=1 to 5)
torchrun --nproc_per_node=8 tools/data_process/main.py navsim \
    --split=trainval_slice_i --gen_meta
```

-----

### nuPlan (10Hz)

The nuPlan dataset provides 10Hz data and is exceptionally large. Based on our tests, processing this requires around 10 nodes (8 GPUs each) running for about a week.

**Step 1: Environment & Preprocessing**
You need to install `nuplan-devkit` and convert nuPlan to `pkl` format using OpenScene's scripts.

```bash
cd third_party/navsim_v1_1
conda env create --name navsim_v1_1 -f environment.yml
conda activate navsim_v1_1
pip install -e .

# Run the preprocessing script. 
# Change 'split=' to mini, trainval, and test inside process.sh to generate 10Hz pkls for all data.
bash tools/data_process/preprocess/nuplan/process.sh
```

**Step 2: Generate Depth Data**

```bash
conda activate dvgt_data

# Process test split
torchrun --nproc_per_node=8 tools/data_process/main.py nuplan \
    --split=test --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image

# Process training slices (run for i=1 to 5)
# Note: By default, data is split into 5 slices. Resuming annotation is not supported.
# If your environment is unstable, increase 'split_quantity' in NuPlanDataset to create more slices.
torchrun --nproc_per_node=8 tools/data_process/main.py nuplan \
    --split=trainval_slice_i \
    --gen_meta --gen_proj_depth --gen_pred_depth --gen_align_depth --gen_image
```

-----

## DVGT2: Anchor Generation

For DVGT2, we need to perform K-means clustering on trajectories and poses to obtain anchors.

During parquet file generation, we already filter out scenarios with abnormal ground truth (GT) poses found in KITTI and OpenScene. Since we use GT pose and generated trajectory truth for supervision, incorrect poses severely impact the model.

**(Optional)** If you are processing your own custom dataset, you can check for abnormal poses first:

```bash
python tools/data_process/anchors/check_outlier_anchors.py
```

**Generate Anchors** (Takes \~15 minutes):

```bash
python tools/data_process/anchors/k_means_anchors.py
```

Alternatively, you can download our pre-generated anchors directly:

```bash
wget https://huggingface.co/datasets/RainyNight/DVGT_demo_dataset/resolve/main/dvgt_2_anchors.zip
unzip dvgt_2_anchors.zip -d data_annotation/anchors/train_total_20
```

-----

## Final Directory Structure

After complete processing, your `data_annotation` directory will be organized as follows (the downloaded demo dataset uses this exact structure):

```text
data_annotation
├── meta
│   └── moge_v2_large_correct_focal
│       └── moge_pred_filter_parquet
│           ├── openscene_train_sample_5.parquet  # Meta info for training
│           └── openscene_test_sample_2.parquet   # Meta info for testing
│
├── image
│   └── moge_v2_large_correct_focal
│       ├── demo                                  # Demo datasets
│       ├── openscene                             # RGB images for OpenScene
│       │   ├── 00001.jpg
│       │   └── ...
│       ├── nuscenes
│       ├── ddad
│       ├── kitti
│       ├── waymo
│       └── argoverse
│
├── proj_depth
│   └── moge_v2_large_correct_focal
│       └── openscene                             # Projected LiDAR depth
│           ├── 00001.png
│           └── ...
│
├── align_depth
│   └── moge_v2_large_correct_focal
│       └── openscene                             # Aligned depth predictions
│           ├── 00001.png
│           └── ...
│
├── pred_depth                                    # Used for data filtering metrics, not training
│   └── moge_v2_large_correct_focal
│       ├── openscene
│       ├── nuscenes
│       ├── ddad
│       ├── kitti
│       └── waymo
│
└── anchors                                       # Generated K-means anchors for DVGT2
    └── train_total_20
        ├── cluster_details.txt
        ├── future_anchors_20_rdf.npz
        ├── future_cluster_mapping_20_rdf.pkl
        ├── past_anchors_20_rdf.npz
        ├── past_cluster_mapping_20_rdf.pkl
        ├── statistics.txt
        ├── vis_future_anchors_20.png
        └── vis_past_anchors_3d_20.png
```