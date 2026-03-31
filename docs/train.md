# Training

This document provides detailed instructions for training and finetuning the DVGT model.

> **💡 Hardware and VRAM Suggestions**
> Our default model was trained on 64 H20 GPUs (146GB VRAM). If you have limited VRAM (e.g., 96GB), we recommend adjusting the following parameters in your configuration file to prevent Out-Of-Memory (OOM) errors:
>
>   * **`accum_steps`**: Increase the number of gradient accumulation steps.
>   * **`max_img_per_gpu`**: Decrease the number of images fed into the model per forward pass (e.g., reducing it to `32` allows for smooth training on a 96GB machine).

-----

## Configuration Overview

We use standard `torchrun` for training, which supports both single-node and multi-node setups. For your convenience, we provide the following ready-to-use configuration files:

**Base Training:**

  * `train/dvgt1/default.yaml`: Full training for DVGT-1.
  * `train/dvgt2/pretrain.yaml`: Pretraining for DVGT-2 (incorporating m-RoPE and local point).
  * `train/dvgt2/default.yaml`: Causal and streaming training for DVGT-2.

**Quick Start / Demo:**

  * `train/dvgt1/demo.yaml`: Quick run for DVGT-1 training using demo data.
  * `train/dvgt2/demo.yaml`: Quick run for DVGT-2 training using demo data.

**Finetuning:**

  * `train/dvgt1/finetune_demo.yaml`: Finetune DVGT-1 on the demo dataset.
  * `train/dvgt2/finetune_demo.yaml`: Finetune DVGT-2 on the demo dataset.
  * `train/dvgt2/default_finetune_navtrain.yaml`: Finetune DVGT-2 on the NavTrain dataset for closed-loop navigation tasks.
  * `train/dvgt2/default_finetune_nuscenes.yaml`: Finetune DVGT-2 on the nuScenes dataset for open-loop navigation tasks.
  * `train/vggt/default.yaml`: Finetune VGGT using absolute metric supervision.

-----

## Run Training

You can choose the appropriate launch command based on your hardware. The examples below demonstrate single-node, multi-GPU training setups (using 8 GPUs). For single-GPU or multi-node cluster scripts, please refer to the [Distributed Training Reference](#distributed-training-reference) section below.

## DVGT-1 Training

### Train with Demo Data

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt1/demo.yaml \
    # Use --debug to enable verbose logging for easier diagnostics and rapid testing.
```

### Train with Full Data

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt1/default.yaml
```

### Finetuning

To finetune the pre-trained DVGT-1 model, first download the model weights (checkpoint) and place them in a local directory, e.g., `ckpt/dvgt1.pt`.

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt1/finetune_demo.yaml \
    checkpoint.direct_load_pretrained_weights_path=ckpt/dvgt1.pt
```

-----

## DVGT-2 Training

### Train directly on Demo Data (Includes Causal and Streaming)

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt2/demo.yaml
```

### Stage 1: Pretraining

Trains a model without causal masking, utilizing m-RoPE, local points, and relative poses.

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt2/pretrain.yaml
```

### Stage 2: Causal and Streaming Training

Builds upon the pre-trained model to train with causal and streaming settings, while simultaneously predicting planning.
*Note: Please refer to the clustering steps in `data_preprocessing.md` to perform clustering before running this stage.*

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt2/default.yaml
```

### Finetuning

To finetune the pre-trained DVGT-2 model, first download the model weights and place them in a local directory, e.g., `ckpt/dvgt2.pt`.

**Finetune on Demo Data:**

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt2/finetune_demo.yaml \
    checkpoint.direct_load_pretrained_weights_path=ckpt/dvgt2.pt
```

**Finetune on nuScenes:** Optimizes performance for open-loop planning.

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt2/default_finetune_nuscenes.yaml
```

**Finetune on NavTrain:** Optimizes performance for closed-loop planning.

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt2/default_finetune_navtrain.yaml
```

-----

## Distributed Training Reference

### 1\. Single-Node, Single-GPU

```bash
torchrun \
    --nproc_per_node=1 \
    tools/train.py \
    --config=train/dvgt1/default.yaml
    # Use --debug to enable verbose logging for easier diagnostics and rapid testing.
```

### 2\. Single-Node, Multi-GPU

```bash
torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    --config=train/dvgt1/default.yaml
```

### 3\. Multi-Node / Cluster

```bash
# Please replace the following environment variables according to your cluster setup:
# $RESOURCE_GPU: Number of GPUs per node
# $WORLD_SIZE: Total number of nodes (servers)
# $RANK: Current node rank (0 to WORLD_SIZE-1)
# $MASTER_ADDR: IP address of the master node
# $MASTER_PORT: Communication port

torchrun \
    --nproc_per_node=${RESOURCE_GPU} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    tools/train.py \
    --config=train/dvgt1/default.yaml
```