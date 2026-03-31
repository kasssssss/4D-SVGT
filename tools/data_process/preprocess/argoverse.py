import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
import concurrent.futures as futures
import multiprocessing as mp

from av2.datasets.sensor.av2_sensor_dataloader import convert_pose_dataframe_to_SE3
from av2.structures.sweep import Sweep
from av2.structures.cuboid import CuboidList, Cuboid
from av2.utils.io import read_feather
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.se3 import SE3
from av2.datasets.sensor.constants import AnnotationCategories

GEN_TRACK = False

PROCESS_SENSOR = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
]

# cam是20hz，lidar是10hz，这个代码以lidar为主，去查询cam的timestamp
# 所以我们对lidar按比例采样即可
original_fps = 10
target_fps = 2
fps_step = original_fps // target_fps

def get_track(scene_source):
    annotations_feather_path = scene_source / "annotations.feather"
    cuboid_list = CuboidList.from_feather(annotations_feather_path)
    raw_data = read_feather(annotations_feather_path)
    ids = raw_data.track_uuid.to_numpy()
    timestamp_cuboid_index = defaultdict(dict)

    for id, cuboid in zip(ids, cuboid_list.cuboids):
        timestamp_cuboid_index[cuboid.timestamp_ns][id] = {
                                                    "rotation": cuboid.dst_SE3_object.rotation,
                                                    "translation":cuboid.dst_SE3_object.translation,
                                                    "length": cuboid.length_m,
                                                    "width": cuboid.width_m,
                                                    "height": cuboid.height_m,
                                                    "category": cuboid.category,
                                                    }
    return timestamp_cuboid_index

import numpy as np
import cv2
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """将四元数转换为旋转矩阵"""
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def build_extrinsic_matrix(extrinsic_row):
    """构建4x4变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(extrinsic_row['qw'], extrinsic_row['qx'], 
                                            extrinsic_row['qy'], extrinsic_row['qz'])
    T[:3, 3] = [extrinsic_row['tx_m'], extrinsic_row['ty_m'], extrinsic_row['tz_m']]
    return T


def build_intrinsic_matrix(intrinsic_row):
    """构建内参矩阵(3x3)"""
    intrinsic = np.eye(3)
    intrinsic[0, 0] = intrinsic_row['fx_px']  # fx
    intrinsic[1, 1] = intrinsic_row['fy_px']  # fy
    intrinsic[0, 2] = intrinsic_row['cx_px']  # cx
    intrinsic[1, 2] = intrinsic_row['cy_px']  # cy
    return intrinsic


def find_closest_timestamp(target_ts, timestamp_list):
    """找到最接近的时间戳"""
    timestamp_array = np.array(timestamp_list)
    idx = np.argmin(np.abs(timestamp_array - target_ts))
    return idx


def project_pointcloud_to_image(lidar_points, camera_intrinsics, camera_extrinsics, image_shape):
    """将激光雷达点云投影到相机图像平面"""
    if lidar_points is None or len(lidar_points) == 0:
        return np.zeros(image_shape[:2], dtype=np.float32)
    
    # 确保点云是齐次坐标 (N, 4)
    if lidar_points.shape[1] == 3:
        points_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
    elif lidar_points.shape[1] == 4:
        # 如果第4列是强度，替换为1
        points_homo = np.hstack([lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))])
    else:
        print(f"点云格式不支持: {lidar_points.shape}")
        return np.zeros(image_shape[:2], dtype=np.float32)
    
    # 转换到相机坐标系
    points_camera = (camera_extrinsics @ points_homo.T).T  # (N, 4)
    
    # 过滤掉相机后面的点 (Z <= 0)
    valid_mask = (points_camera[:, 2] > 1) & (points_camera[:, 2] < 150)  # 不在这个范围的lidar都认为不可靠
    points_camera = points_camera[valid_mask]
    
    if len(points_camera) == 0:
        return np.zeros(image_shape[:2], dtype=np.float32)
    
    # 构建相机内参矩阵
    K = camera_intrinsics
    
    # 投影到图像平面
    points_2d = (K @ points_camera[:, :3].T).T  # (N, 3)
    points_2d = points_2d / points_2d[:, 2:3]  # 齐次坐标归一化
    
    # 提取像素坐标和深度
    u = points_2d[:, 0].round().astype(int)
    v = points_2d[:, 1].round().astype(int)
    depths = points_camera[:, 2]  # 深度值 (Z坐标)
    
    # 过滤超出图像边界的点
    h, w = image_shape
    valid_pixels = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depths > 0)
    u = u[valid_pixels]
    v = v[valid_pixels]
    depths = depths[valid_pixels]
    
    # 创建深度图
    depth_map = np.zeros((h, w), dtype=np.float32)
    depth_map[v, u] = depths
    
    return depth_map

def depth_merge_to_rgb(depth_map, rgb_image, is_dilate=False):
    """将深度图与RGB图像融合"""
    if depth_map is None or rgb_image is None:
        return None
    
    # 确保深度图和RGB图像的尺寸一致
    if depth_map.shape != rgb_image.shape[:2]:
        print("深度图和RGB图像尺寸不一致")
        return None
    
    if is_dilate is True:
        # 膨胀深度图以增强深度感
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        depth_map = cv2.dilate(depth_map, kernel)

    # 融合深度图和RGB图像
    depth_normalized = cv2.normalize(depth_map.round(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    alpha = 0.5
    mask = depth_map > 0
    # 仅在深度有效区域显示颜色
    overlay = rgb_image.copy()
    overlay[mask] = cv2.addWeighted(overlay[mask], 1 - alpha, color_depth[mask], alpha, 0)
    return overlay

def process_scene(scene_source, scene_target, scene_name):
    """处理单个场景"""

    # 读取内外参数据
    intrinsic_path = scene_source / "calibration" / "intrinsics.feather"
    sensor_to_ego_path = scene_source / "calibration" / "egovehicle_SE3_sensor.feather"

    intrinsics_df = pd.read_feather(intrinsic_path)
    sensor_to_ego_df = pd.read_feather(sensor_to_ego_path)

    # 获取所有传感器名称
    camera_sensors = [f.name for f in (scene_source / "sensors" / "cameras").iterdir() if f.is_dir() and f.name in PROCESS_SENSOR]

    # 处理每个相机传感器
    img_flie_ori = {}
    image_files_clean = {}
    image_timestamps = {}
    image_timestamps_compare = {}
    intrinsic_matrixes = {}
    sensor_to_ego_matrixes = {}
    image_sizes = {}

    for sensor in camera_sensors:

        # 获取该传感器的内外参
        sensor_intrinsic = intrinsics_df[intrinsics_df['sensor_name'] == sensor]
        sensor_to_ego = sensor_to_ego_df[sensor_to_ego_df['sensor_name'] == sensor]

        if len(sensor_intrinsic) == 0 or len(sensor_to_ego) == 0:
            continue

        # 构建矩阵
        intrinsic_matrixes[sensor] = build_intrinsic_matrix(sensor_intrinsic.iloc[0])
        sensor_to_ego_matrixes[sensor] = build_extrinsic_matrix(sensor_to_ego.iloc[0])

        # 图像尺寸
        image_sizes[sensor] = (int(sensor_intrinsic.iloc[0]['height_px']), 
                    int(sensor_intrinsic.iloc[0]['width_px']))

        # 获取所有图像文件
        sensor_path = (scene_source / "sensors" / "cameras" / sensor)
        img_flie_ori[sensor] = [(scene_source / "sensors" / "cameras" / sensor / img) for img in os.listdir(str(sensor_path))]
        image_timestamps[sensor] = [int(f.name.split('.')[0]) for f in img_flie_ori[sensor]]
        image_timestamps_compare[sensor] = [int(f.name[:12]) for f in img_flie_ori[sensor]]


    # 获取 pose 文件
    log_poses_df = read_feather(scene_source / "city_SE3_egovehicle.feather")

    # 获取全场景track
    if GEN_TRACK:
        all_track_data = get_track(scene_source)

    # 获取所有点云文件
    lidar_files = list((scene_source / "sensors" / "lidar").glob("*.feather"))
    lidar_files = sorted(lidar_files)[::fps_step]   # 我们只标注2hz的数据
    lidar_timestamps = [int(f.stem) for f in lidar_files]
    lidar_timestamps_compare = [int(f.stem[:12]) for f in lidar_files]

    for lidar_file, lidar_ts, lidar_ts_cmp in zip(lidar_files, lidar_timestamps, lidar_timestamps_compare):

        # 读取点云数据
        lidar_path = scene_source / "sensors" / "lidar" / f"{lidar_ts}.feather"
        pointcloud_df = pd.read_feather(lidar_path)
        points = pointcloud_df[['x', 'y', 'z']].values

        pose = convert_pose_dataframe_to_SE3(log_poses_df.loc[log_poses_df["timestamp_ns"] == lidar_ts])            # eog -> world
        for sensor in camera_sensors:
            closest_cam_idx = find_closest_timestamp(lidar_ts_cmp, image_timestamps_compare[sensor])

            img_ts = image_timestamps[sensor][closest_cam_idx]
            img_file = img_flie_ori[sensor][closest_cam_idx]

            intrinsic_matrix = intrinsic_matrixes[sensor]
            sensor_to_ego = sensor_to_ego_matrixes[sensor]
            sensor_to_world_matrix = pose.transform_matrix @ sensor_to_ego
            world_to_cam = np.linalg.inv(sensor_to_world_matrix)
            image_size = image_sizes[sensor]

            # 生成深度图
            depth_map = project_pointcloud_to_image(
                points, intrinsic_matrix, np.linalg.inv(sensor_to_ego), image_size
            )

            # 创建目标目录
            jpg_dir = scene_target / "jpg" / sensor
            depth_dir = scene_target / "depth" / sensor
            # depth_vis_dir = scene_target / "depth_vis" / sensor
            param_dir = scene_target / "meta_parameters" / sensor


            jpg_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            # depth_vis_dir.mkdir(parents=True, exist_ok=True)
            param_dir.mkdir(parents=True, exist_ok=True)

            # 保存图像
            try:
                jpg_target = jpg_dir / f"{img_ts}.jpg"
                shutil.copy2(img_file, jpg_target)
            except:
                import pdb; pdb.set_trace()


            # 保存深度图
            depth_target = depth_dir / f"{img_ts}_depth.npy"
            np.save(depth_target, depth_map)

            # depth_vis_path = depth_vis_dir / f"{img_ts}_depth.png"
            # depth_vis = depth_merge_to_rgb(depth_map, cv2.imread(img_file), is_dilate=True)
            # cv2.imwrite(depth_vis_path, depth_vis)

            # 保存meta信息
            param_target = param_dir / f"{img_ts}_cam.npz"
            np.savez(
                param_target, 
                world_to_cam=world_to_cam,
                camera_intrinsics=intrinsic_matrix,     # cam is rdf
                ego_to_world=pose.transform_matrix,      # ego and world is flu
            )

        # 保存轨迹数据
        if GEN_TRACK:
            track_data = all_track_data[lidar_ts]
            track_dir = scene_target / "tracks"
            track_dir.mkdir(parents=True, exist_ok=True)
            track_file = track_dir / f"{lidar_ts}.pkl"
            with open(track_file, 'wb') as f:
                pickle.dump(track_data, f)

def worker(args):
    """
    args = (scene_folder, target_root, idx)
    """
    scene_folder, target_root, idx = args
    scene_name = scene_folder.name
    scene_target = Path(target_root) / scene_name

    process_scene(scene_folder, scene_target, scene_name)
    return scene_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Root directory of argoverse2 dataset (e.g., path/to/sensor).")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory of extracted data.")
    args = parser.parse_args()

    # 定义需要处理的数据集划分
    splits = ["train", "val", "test"]
    
    # 确定最大进程数
    max_workers = 16
    max_workers = min(max_workers, mp.cpu_count()) 

    for split in splits:
        print(f"\n========== 开始处理 {split} 集 ==========")
        
        # 动态拼接路径
        source_path = Path(args.input_dir) / split
        target_path = Path(args.output_dir) / split

        # 检查源路径是否存在，避免因为缺少某个划分导致脚本崩溃
        if not source_path.exists():
            print(f"警告：未找到路径 {source_path}，跳过该划分。")
            continue

        # 确保目标文件夹存在
        target_path.mkdir(parents=True, exist_ok=True)

        scene_folders = [f for f in source_path.iterdir() if f.is_dir()]
        
        if not scene_folders:
            print(f"警告：{source_path} 目录下为空，跳过。")
            continue

        tasks = [(scene, target_path, i) for i, scene in enumerate(scene_folders)]
        
        # debug
        # worker(tasks[0])
        # 启动多进程处理
        
        with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            # tqdm 显示当前划分的进度
            list(tqdm(pool.map(worker, tasks),
                      total=len(tasks),
                      desc=f"多进程处理 {split} 场景"))

    print("\n所有数据集转换完成！")