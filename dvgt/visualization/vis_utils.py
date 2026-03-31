from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import *
from time import sleep
import viser
import viser.transforms as viser_tf
from typing import Optional, List

"""
from dvgt.visualization.vis_utils import visualize_depth, visualize_scene, \
    save_rgb_image, visual_cam_pose_in_viser, visual_point3d_in_viser
"""

def visualize_depth(depth_array: np.ndarray, output_filename: str, max_depth: float = 200):
    valid_mask = np.isfinite(depth_array)
    valid_depths = depth_array[valid_mask]

    # 确保可视化的depth在同一个尺度
    min_val = 0
    max_val = max_depth if max_depth > 0 else valid_depths.max()
    depth_array = depth_array.clip(min_val, max_val)
    normalized_depth = np.zeros_like(depth_array, dtype=np.float32)
    normalized_depth[valid_mask] = (valid_depths - min_val) / (max_val - min_val)
    
    colormap = plt.get_cmap('viridis')
    colored_depth_rgba = colormap(normalized_depth)
    colored_depth_rgb = (colored_depth_rgba[:, :, :3] * 255).astype(np.uint8)
    
    # 将无效区域（INF或NaN）的像素设置为白色
    colored_depth_rgb[~valid_mask] = [255, 255, 255]

    final_image_bgr = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filename, final_image_bgr)

def save_rgb_image(image_array: np.ndarray, path: str):
    # 3, H, W
    if image_array.shape[0] == 3:
        image_array = image_array.transpose(1, 2, 0)

    if image_array.max() <= 1:
        image_array = image_array * 255

    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    image = Image.fromarray(image_array)
    image.save(path)

def visual_point3d_in_viser(    
    point3d: np.ndarray,    # [..., H, W, 3]
    colors: np.ndarray,     # [..., H, W, 3]
    mask: Optional[np.ndarray] = None, 
    downsample_ratio: float = 1.0,
    point_size: float = 0.05,
):
    if mask is None:
        mask = np.isfinite(point3d).all(-1)
    point3d = point3d[mask]
    colors = colors[mask]
    
    # 随机下采样
    num_points = point3d.shape[0]
    target_num_points = int(num_points * downsample_ratio)
    indices = np.random.choice(num_points, target_num_points, replace=False)
    point3d = point3d[indices]
    colors = colors[indices]
    print(f"下采样后剩余 {point3d.shape[0]} 个点。")

    # 将点云中心移至原点，便于观察
    points_center = point3d.mean(axis=0)
    point3d -= points_center
    print(f"点云已居中，中心点: {points_center}")

    server = viser.ViserServer()
    server.scene.add_point_cloud(
        name="point_cloud_from_parquet",
        points=point3d, 
        colors=colors,
        point_size=point_size,
    )
    print("可视化服务器已启动。请在浏览器中打开提供的链接。")

    idx = 0
    while idx < 30:
        idx += 1
        sleep(1)

def visual_cam_pose_in_viser(
    cam_to_world: np.ndarray,
    intrinsics: Optional[np.ndarray] = None,
    images: Optional[np.ndarray] = None,
    port: int = 8080,
    frustum_scale: float = 0.1,
    axes_length: float = 0.05,
    axes_radius: float = 0.002,
) -> None:
    """
    使用 Viser 启动一个服务器来可视化一组相机位姿（坐标系和视锥）。
    【简化版：无点击功能】

    Args:
        cam_to_world (np.ndarray):
            相机外参 (S, 3, 4) 或 (S, 4, 4)，表示从相机到世界的变换。
        images (Optional[np.ndarray], optional):
            要显示在视锥中的图像 (S, H, W, 3)，格式为 RGB, uint8。
        intrinsics (Optional[np.ndarray], optional):
            相机内参 (S, 3, 3)，用于计算精确的FOV。
        port (int, optional): Viser 服务器端口。
        frustum_scale (float, optional): 视锥的大小。
        axes_length (float, optional): 坐标轴的长度。
        axes_radius (float, optional): 坐标轴的半径。
    """

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    print(f"Viser server started at http://0.0.0.0:{port}")

    # --- 1. 验证和准备输入 ---
    if not (cam_to_world.ndim == 3 and cam_to_world.shape[2] == 4 and cam_to_world.shape[1] >= 3):
        raise ValueError(
            f"cam_to_world 必须是 (S, 3, 4) 或 (S, 4, 4) 形状, "
            f"但得到 {cam_to_world.shape}"
        )

    S = cam_to_world.shape[0]
    cam2world_mats = cam_to_world[:, :3, :]

    if intrinsics is not None and intrinsics.shape != (S, 3, 3):
        raise ValueError(f"intrinsics 必须是 (S, 3, 3) 形状, 但得到 {intrinsics.shape}")

    image_list: List[Optional[np.ndarray]] = [None] * S
    default_h, default_w = 480, 640

    if images is not None:
        if images.shape[0] != S:
            raise ValueError(f"外参数量 ({S}) 和图像数量 ({images.shape[0]}) 不匹配")
        if images.shape[-1] != 3:
            raise ValueError(f"图像必须是 (S, H, W, 3) 格式, 但得到 {images.shape}")
        if images.dtype != np.uint8:
            if images.max() <= 1.0 and images.min() >= 0.0:
                images = (images * 255).astype(np.uint8)
            else:
                images = images.astype(np.uint8)
        image_list = [img for img in images]
        default_h, default_w = image_list[0].shape[:2]

    # --- 2. 场景居中 (推荐) ---
    positions = cam2world_mats[:, :, 3]
    scene_center = np.mean(positions, axis=0)
    cam2world_mats_centered = cam2world_mats.copy()
    cam2world_mats_centered[:, :, 3] -= scene_center

    server.scene.add_frame("world_origin", axes_length=axes_length * 2, axes_radius=axes_radius * 2)

    # --- 3. 循环添加所有相机 ---
    for i in range(S):
        cam2world_3x4 = cam2world_mats_centered[i]
        T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

        frame_name = f"frame_{i}"

        # 添加相机坐标系
        server.scene.add_frame(
            frame_name,
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=axes_length,
            axes_radius=axes_radius,
        )

        image = image_list[i]
        h, w = (image.shape[:2] if image is not None else (default_h, default_w))
        aspect = w / h

        # 计算 FOV
        if intrinsics is not None:
            fy = intrinsics[i, 1, 1]
            fov = 2 * np.arctan2(h / 2, fy)
        else:
            default_fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, default_fy)

        # 添加相机视锥
        server.scene.add_camera_frustum(
            f"{frame_name}/frustum",
            fov=fov,
            aspect=aspect,
            scale=frustum_scale,
            image=image,
            line_width=1.0
        )

    idx = 0
    while idx < 30:
        idx += 1
        sleep(1)

def visualize_scene(
    # --- Point Cloud Args ---
    point3d: Optional[np.ndarray] = None,    # Shape: [..., H, W, 3]
    colors: Optional[np.ndarray] = None,     # Shape: [..., H, W, 3], range [0, 255]
    point_mask: Optional[np.ndarray] = None, # Shape: [..., H, W]
    downsample_ratio: float = 1.0,
    point_size: float = 0.01,
    
    # --- Camera Pose Args ---
    cam_to_world: Optional[np.ndarray] = None, # Shape: (S, 3, 4) or (S, 4, 4)
    intrinsics: Optional[np.ndarray] = None,   # Shape: (S, 3, 3)
    images: Optional[np.ndarray] = None,       # Shape: (S, H, W, 3), RGB uint8
    frustum_scale: float = 0.5,
    axes_length: float = 0.5,
    axes_radius: float = 0.02,

    # --- Server Args ---
    port: int = 8080,
) -> None:
    """
    使用 Viser 启动一个服务器，统一可视化3D点云和/或相机位姿。

    可以单独可视化点云，单独可视化相机，或将它们在同一场景中进行可视化。
    场景会根据输入进行居中，优先以相机位姿的中心为准。

    Args:
        point3d (Optional[np.ndarray]): 3D点云坐标，形状为 [..., H, W, 3]。
        colors (Optional[np.ndarray]): 点云颜色，形状与 point3d 匹配，值为 0-255。
        point_mask (Optional[np.ndarray]): 用于过滤点云的布尔掩码。
        downsample_ratio (float): 点云随机下采样比率，1.0 表示不下采样。
        point_size (float): 可视化时每个点的大小。
        
        cam_to_world (Optional[np.ndarray]): 相机外参 (S, 3, 4) 或 (S, 4, 4)，从相机到世界坐标系的变换。
        intrinsics (Optional[np.ndarray]): 相机内参 (S, 3, 3)，用于计算精确的FOV。
        images (Optional[np.ndarray]): 要显示在视锥中的图像 (S, H, W, 3)，RGB uint8格式。
        frustum_scale (float): 相机视锥的大小。
        axes_length (float): 坐标轴的长度。
        axes_radius (float): 坐标轴的半径。

        port (int): Viser 服务器运行的端口。
    """
    if point3d is None and cam_to_world is None:
        print("没有提供点云或相机位姿，无可視化内容。")
        return

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    print(f"Viser server started at http://0.0.0.0:{port}")

    scene_center = np.zeros(3)
    
    # --- 1. 预处理和计算场景中心 ---
    processed_points = None
    processed_colors = None

    # 优先使用相机位姿来确定场景中心
    if cam_to_world is not None:
        if not (cam_to_world.ndim == 3 and cam_to_world.shape[2] == 4 and cam_to_world.shape[1] >= 3):
            raise ValueError(f"cam_to_world 必须是 (S, 3, 4) 或 (S, 4, 4) 形状, 但得到 {cam_to_world.shape}")
        positions = cam_to_world[:, :3, 3]
        scene_center = np.mean(positions, axis=0)
        print(f"场景已根据相机位姿居中，中心点: {scene_center}")
    
    # 如果有点云，处理点云数据
    if point3d is not None:
        if colors is None:
            raise ValueError("提供了 point3d，但未提供 colors。")

        if point_mask is None:
            point_mask = np.isfinite(point3d).all(-1)
        
        processed_points = point3d[point_mask]
        processed_colors = colors[point_mask]
        
        # 如果没有相机位姿，则使用点云来确定场景中心
        if cam_to_world is None:
            scene_center = processed_points.mean(axis=0)
            print(f"场景已根据点云居中，中心点: {scene_center}")

        # 下采样
        if downsample_ratio < 1.0:
            num_points = processed_points.shape[0]
            target_num_points = int(num_points * downsample_ratio)
            indices = np.random.choice(num_points, target_num_points, replace=False)
            processed_points = processed_points[indices]
            processed_colors = processed_colors[indices]
            print(f"点云下采样后剩余 {processed_points.shape[0]} 个点。")

    # --- 2. 添加可视化元素到场景 ---

    # 添加相机位姿
    if cam_to_world is not None:
        S = cam_to_world.shape[0]
        cam2world_mats = cam_to_world[:, :3, :]
        cam2world_mats_centered = cam2world_mats.copy()
        cam2world_mats_centered[:, :, 3] -= scene_center

        image_list: List[Optional[np.ndarray]] = [None] * S
        default_h, default_w = 480, 640

        if images is not None:
            if images.shape[0] != S: raise ValueError(f"外参数量 ({S}) 和图像数量 ({images.shape[0]}) 不匹配")
            image_list = [img for img in images]
            default_h, default_w = image_list[0].shape[:2]

        # server.scene.add_frame("world_origin", axes_length=axes_length * 2, axes_radius=axes_radius * 2)

        for i in range(S):
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_mats_centered[i])
            frame_name = f"/camera/{i}"
            server.scene.add_frame(
                frame_name, wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=axes_length, axes_radius=axes_radius
            )
            image = image_list[i]
            h, w = (image.shape[:2] if image is not None else (default_h, default_w))
            aspect = w / h
            if intrinsics is not None:
                fy = intrinsics[i, 1, 1]
                fov = 2 * np.arctan2(h / 2, fy)
            else:
                default_fy = 1.1 * h
                fov = 2 * np.arctan2(h / 2, default_fy)
            
            server.scene.add_camera_frustum(
                f"{frame_name}/frustum", fov=fov, aspect=aspect,
                scale=frustum_scale, image=image, line_width=1.0
            )
        print(f"已添加 {S} 个相机位姿到场景。")

    # 添加点云
    if processed_points is not None:
        # 将点云中心移至原点
        processed_points -= scene_center
        server.scene.add_point_cloud(
            name="/point_cloud",
            points=processed_points,
            colors=processed_colors,
            point_size=point_size,
        )
        print("已添加点云到场景。")

    idx = 0
    while idx < 30:
        idx += 1
        sleep(1)

def visualize_depth_scatter_plot(pred_depths, lidar_depths, filepath):

    valid_mask = (lidar_depths > 0) & (pred_depths > 0) & \
                np.isfinite(pred_depths) & np.isfinite(lidar_depths)
    pred_flat = pred_depths[valid_mask]
    lidar_flat = lidar_depths[valid_mask]

    # 随机采样一部分数据来绘图，以防数据量过大导致卡顿
    sample_indices = np.random.choice(len(pred_flat), size=min(100000, len(pred_flat)), replace=False)

    plt.figure(figsize=(10, 8))
    plt.scatter(pred_flat[sample_indices], lidar_flat[sample_indices], s=1, alpha=0.5)
    plt.xlabel("Predicted Depth (Flattened)")
    plt.ylabel("LiDAR Depth (Flattened)")
    plt.title("Scatter Plot of Predicted vs. LiDAR Depth")
    plt.grid(True)
    plt.savefig(filepath)