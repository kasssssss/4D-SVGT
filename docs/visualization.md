# Visualization Tools

Our visualization toolkit is built on top of `viser` to provide interactive and high-quality 3D rendering. We offer various visualization pipelines ranging from basic point cloud inspection to full-scene automated demos and debugging utilities.

## Table of Contents
- [1. Basic Point Cloud Visualization](#1-basic-point-cloud-visualization)
- [2. DVGT1 Demos](#2-dvgt1-demos)
- [3. DVGT2 Demos](#3-dvgt2-demos)
- [4. Debugging & Interactive Visualization Utilities](#4-debugging--interactive-visualization-utilities)

---

## 1. Basic Point Cloud Visualization

To inspect standard point clouds, you must first extract the scene data into a `.pkl` file, and then launch the visualization server.

### Step 1: Extract and Save Point Cloud Data
Run the following command to extract the point map:
```bash
torchrun --nproc_per_node=1 dvgt/visualization/save_pointmap.py
```
> **Note:** You can specify which scene to save by modifying the `target_keys` variable inside `visual_dataset` based on your desired scene ID.

### Step 2: Launch the Visualization
Once the `.pkl` file is saved, you can use `visual.py` to inspect the data. Below are several usage examples (check the script's arguments for more detailed controls):

**Visualize Predicted Point Cloud:**
```bash
python dvgt/visualization/visual.py \
    --pkl_path=path/to/saved_data.pkl \
    --mode=pred \
    --conf_percentile=30 \
    --save_info
```

**Visualize the Difference (Error) Between Pred and GT:**
```bash
python dvgt/visualization/visual.py \
    --pkl_path=path/to/saved_data.pkl \
    --mode=pred \
    --error_threshold=10
```

**Visualize Ground Truth (GT) Point Cloud:**
```bash
python dvgt/visualization/visual.py \
    --pkl_path=path/to/saved_data.pkl \
    --mode=gt \
    --save_info
```

---

## 2. DVGT1 Demos

These demos provide automated camera movements through a generated scene.

**Prerequisite:** You must first extract the scene data using the point map saving script to get your `.pkl` file:
```bash
torchrun --nproc_per_node=1 dvgt/visualization/save_pointmap.py
```

**Ego-Vehicle Trajectory Demo:**
Automatically plays an animation that follows the ego-vehicle's forward trajectory through the entire scene.
```bash
python dvgt/visualization/dvgt1_demo/camera_move_forward.py
```

**Scene Orbit Demo:**
Automatically orbits the camera around the entire 3D point cloud scene.
```bash
python dvgt/visualization/dvgt1_demo/camera_move_orbit.py
```

---

## 3. DVGT2 Demos

The DVGT2 demos showcase longer sequences with ego-vehicle integration. This requires a bit more preparation to ensure you have sufficiently long scenes.

### 3.1. Prerequisites

**1. Download the Car Model:**
Download the 3D car asset (Xiaomi SU7) and place it in the designated directory:
```bash
wget https://huggingface.co/datasets/RainyNight/DVGT_demo_dataset/resolve/main/xiaomi_yu7.pkl -O dvgt/visualization/dvgt2_demo/xiaomi_yu7.pkl
```

**2. Prepare Long Sequence Data:**
Standard OpenScene logs are typically limited to 40 frames. For this demo, you need longer continuous sequences. Please refer to the [NavSim section in docs/data_preparation.md](docs/data_preparation.md#navsim) to generate the `navtrain` parquet files.
> **Tip:** You can filter for long scenes by adjusting the `MIN_FRAMES_THRESHOLD` parameter in `tools/data_process/postprocess/downsample_parquet_data.py`.

### 3.2. DVGT2 Execution Pipeline
To run the full DVGT2 demo, execute the following scripts in order:
1. **Extract Data:** Save the initial scene point map.
   ```bash
   torchrun --nproc_per_node=1 dvgt/visualization/save_pointmap.py
   ```
   
2. **Post-Process Data:** Process the generated `.pkl` specifically for the DVGT2 demo.
   ```bash
   python dvgt/visualization/dvgt2_demo/post_process_pkl.py
   ```

3. **Prepare Images for SAM3:** To properly visualize the scene (accumulating static background objects while only rendering dynamic objects frame-by-frame), you need to segment the dynamic objects. First, format and save the images to the target directory for SAM inference.
   ```bash
   python dvgt/visualization/dvgt2_demo/save_images_for_sam.py
   ```

4. **Run SAM3 Segmentation:** Execute the segmentation script to isolate the dynamic objects.
   ```bash
   python dvgt/visualization/dvgt2_demo/sam3_seg.py
   ```

5. **Run Demo:** Launch the interactive DVGT2 demo with trajectories, segmented dynamic objects, and the 3D vehicle model.
   ```bash
   python dvgt/visualization/dvgt2_demo/visual.py
   ```


### 3.3. Optional Features

* **Traditional IPM Visualization:** We also provide a script to generate direct projection images using the traditional Inverse Perspective Mapping (IPM) method.
  ```bash
  python dvgt/visualization/dvgt2_demo/visual_ipm.py
  ```

---

## 4. Debugging & Interactive Visualization Utilities

This module provides a collection of quick-to-use visualization functions designed for debugging depth maps, RGB images, 3D point clouds, and camera poses. It relies on `OpenCV`, `Matplotlib`, and `viser` for interactive 3D rendering.

### How to Import
You can quickly import the specific functions you need anywhere in your code during debugging:

```python
from dvgt.visualization.vis_utils import (
    visualize_depth, 
    visualize_scene, 
    save_rgb_image, 
    visual_cam_pose_in_viser, 
    visual_point3d_in_viser,
    visualize_depth_scatter_plot
)
```

### Function Overview

#### 2D Image & Depth Utilities
* **`visualize_depth(depth_array, output_filename, max_depth=200)`**
    Converts a 2D depth array into a color-coded image (using the `viridis` colormap) and saves it to disk. It automatically handles invalid values (NaN/Inf) by rendering them as white pixels.
* **`save_rgb_image(image_array, path)`**
    A robust utility to save numpy arrays as RGB images. It automatically handles shape formats (converts `CHW` to `HWC`) and scales float values (`0-1`) to `uint8` (`0-255`) before saving.
* **`visualize_depth_scatter_plot(pred_depths, lidar_depths, filepath)`**
    Generates and saves a scatter plot comparing predicted depth against ground-truth LiDAR depth. It randomly samples up to 100,000 points to ensure fast rendering without memory overload.

#### 3D Interactive Visualization (Powered by `viser`)
*These functions launch a local web server (default port 8080). After calling them, open the provided `http://0.0.0.0:8080` link in your browser to interact with the 3D scene.*

* **`visual_point3d_in_viser(point3d, colors, ...)`**
    Visualizes an isolated 3D point cloud. It automatically centers the points at the origin for easier viewing and supports random downsampling for massive point clouds.
* **`visual_cam_pose_in_viser(cam_to_world, intrinsics, images, ...)`**
    Visualizes camera trajectories in 3D space. It renders camera axes and frustums based on camera extrinsics and intrinsics. You can optionally project RGB images onto the camera frustums.
* **`visualize_scene(point3d, colors, cam_to_world, ...)`**
    A unified function to visualize **both** 3D point clouds and camera poses together in the same interactive scene. It automatically aligns and centers the entire scene based on the camera trajectory.