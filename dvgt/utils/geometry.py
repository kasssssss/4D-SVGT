# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple, Union

def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            World coordinates (H, W, 3)
            cam coordinates (H, W, 3)
            valid depth mask (H, W)
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic)

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    以闭式解计算一批SE(3)矩阵的逆。
    这个函数支持任意批次维度 (...) 以及单个矩阵。

    Args:
        se3: SE(3) 变换矩阵，形状为 (..., 4, 4) 或 (..., 3, 4)。

    Returns:
        逆变换矩阵，与输入 se3 的类型、设备和批次维度相同，形状为 (..., 4, 4)。
        
    公式:
        如果 T_mat = [[R, t], [0, 1]]
        则 T_mat_inv = [[R.T, -R.T @ t], [0, 1]]
    """
    # 检查输入是 NumPy 数组还是 PyTorch 张量
    is_numpy = isinstance(se3, np.ndarray)
    
    # 为单个矩阵 (ndim=2) 临时增加一个批次维度，以便统一处理
    was_single = se3.ndim == 2
    if was_single:
        se3 = se3[None, ...]  # works for both numpy and torch

    # 验证最后两个维度
    if se3.shape[-2:] not in [(4, 4), (3, 4)]:
        raise ValueError(f"输入矩阵的最后两个维度必须是 (4, 4) 或 (3, 4)，但得到的是 {se3.shape}")

    R = se3[..., :3, :3]
    t = se3[..., :3, 3:4]

    if is_numpy:
        R_transposed = np.swapaxes(R, -2, -1)
    else:
        R_transposed = R.transpose(-2, -1)
    
    t_inverted = -R_transposed @ t

    batch_shape = se3.shape[:-2]
    
    if is_numpy:
        inverted_matrix = np.zeros((*batch_shape, 4, 4), dtype=se3.dtype)
    else:
        inverted_matrix = torch.zeros((*batch_shape, 4, 4), device=se3.device, dtype=se3.dtype)
    
    inverted_matrix[..., :3, :3] = R_transposed
    inverted_matrix[..., :3, 3:4] = t_inverted
    inverted_matrix[..., 3, 3] = 1.0

    # 5. 如果输入是单个矩阵，移除之前添加的批次维度
    if was_single:
        inverted_matrix = inverted_matrix[0]

    return inverted_matrix

def to_homogeneous(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    将一个 (..., 3, 4) 的变换矩阵批量转换为 (..., 4, 4) 的齐次矩阵。
    """
    is_torch = isinstance(matrix, torch.Tensor)

    batch_dims = matrix.shape[:-2]
    
    if is_torch:
        last_row_expanded = torch.zeros((*batch_dims, 1, 4), dtype=matrix.dtype, device=matrix.device)
        last_row_expanded[..., 3] = 1.0
        return torch.cat([matrix, last_row_expanded], dim=-2)
    else:
        last_row_expanded = np.zeros((*batch_dims, 1, 4), dtype=matrix.dtype)
        last_row_expanded[..., 3] = 1.0
        return np.concatenate([matrix, last_row_expanded], axis=-2)

def transform_ego_pose_point3d_to_first_ego(
    world_points: np.ndarray,
    ego_to_worlds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将ego to world和point3d转到第一个ego为中心的坐标系下
        
    Args:
        world_points: 3D points in world coordinates of shape (T, V, H, W, 3)
        ego_to_worlds: Transformation matrix from ego to world of shape (T, 4, 4)
    """
    # 将ego to worlds转化到第一帧ego为中心的坐标系下，得到：ego_N to ego_1
    world_to_ego_first = closed_form_inverse_se3(ego_to_worlds[0])[None]     # (1 4 4)
    ego_N_to_ego_first = world_to_ego_first @ ego_to_worlds         # (T 4 4)

    # 将世界坐标系的point3d，转换到ego_first为中心的坐标系下
    world_to_ego_first = world_to_ego_first[:, None, None]     # (T V H 4 4)
    R = world_to_ego_first[..., :3, :3]                                   # (T V H 3 3)
    t = world_to_ego_first[..., :3, 3]                                    # (T V H 3)
    points_in_ego_first = (world_points @ R.swapaxes(-1, -2)) + t[..., None, :]
    
    return ego_N_to_ego_first.astype(np.float32), points_in_ego_first.astype(np.float32)

def transform_point3d_in_world_to_ego_n(
    world_points: np.ndarray,
    ego_to_worlds: np.ndarray,
) -> np.ndarray:
    """
    将world_points 转到 每一帧ego为中心的坐标系
        
    Args:
        world_points: 3D points in world coordinates of shape (T, V, H, W, 3)
        ego_to_worlds: Transformation matrix from ego to world of shape (T, 4, 4)   
    """
    worlds_to_ego = closed_form_inverse_se3(ego_to_worlds)   # (T 4 4)
    R_worlds_to_ego = worlds_to_ego[..., :3, :3]
    t_worlds_to_ego = worlds_to_ego[..., :3, 3]   # (T 3)

    # 将世界坐标系的point3d，转换到每一帧ego为中心的坐标系下
    points_in_ego = world_points @ R_worlds_to_ego.swapaxes(-1, -2)[:, None, None] + t_worlds_to_ego[:, None, None, None]

    return points_in_ego.astype(np.float32)

def transform_extrinsics_to_first_ego(
    world_to_cam: np.ndarray,
    ego_to_world: np.ndarray,
) -> np.ndarray:
    """
    将world to cam转到为ego first to cam，注意，输入的world坐标系，应该是float64
        
    Args:
        world_to_cam: shape (T, V, 4, 4)
        ego_to_world: shape (T, 4, 4)   
    return: shape (T, V, 4, 4) 
    """
    ego_first_to_world = ego_to_world[0:1, None]

    ego_first_to_cam = world_to_cam @ ego_first_to_world

    return ego_first_to_cam.astype(np.float32)

def transform_T_cam_n_ego_first_TO_T_cam_n_cam_first(
    T_cam_n_ego_first: np.ndarray,  # [T, V, 4, 4]
) -> np.ndarray:

    T_cam_first_ego_first = T_cam_n_ego_first[0:1, 0:1]
    T_ego_first_cam_first = closed_form_inverse_se3(T_cam_first_ego_first)

    T_cam_n_cam_first = T_cam_n_ego_first @ T_ego_first_cam_first

    return T_cam_n_cam_first

def transform_points_in_ego_first_to_cam_first(
    points_in_ego_first: np.ndarray,    # [T, V, H, W, 3]
    ego_first_to_cam_n: np.ndarray,      # [T, V, 4, 4]
) -> np.ndarray:
    """
    return: points_in_cam_first: [T, V, H, W, 3]
    """
    ego_first_to_cam_first = ego_first_to_cam_n[0, 0][None, None, None]     # [1, 1, 1, 4, 4]
    R = ego_first_to_cam_first[..., :3, :3]
    t = ego_first_to_cam_first[..., :3, 3]

    points_in_cam_first = points_in_ego_first @ R.swapaxes(-1, -2) + t[..., None, :]

    return points_in_cam_first

def get_relative_future_poses(ego_to_worlds, future_ego_to_worlds):
    """
    计算每个时刻相对于当前时刻的未来轨迹变换矩阵。
    注意，这里涉及世界坐标系，必须使用float64计算
    
    Args:
        ego_to_worlds: [T, 4, 4], 历史/当前的位姿
        future_ego_to_worlds: [T_future, 4, 4], 接续的未来位姿
        
    Returns:
        relative_poses: [T, T_future, 4, 4]
    """
    T = ego_to_worlds.shape[0]
    T_future = future_ego_to_worlds.shape[0]    
    all_poses = np.concatenate([ego_to_worlds, future_ego_to_worlds], axis=0)

    # 提取需要的未来帧序列
    range_t = np.arange(T)
    offsets = np.arange(1, T_future + 1)[None, :]
    target_indices = range_t[:, None] + offsets
    target_poses = all_poses[target_indices]
    
    # 转换到当前帧坐标系下
    worlds_to_ego_batch = closed_form_inverse_se3(ego_to_worlds)
    relative_poses = worlds_to_ego_batch[:, None] @ target_poses
        
    return relative_poses.astype(np.float32)

def compute_ego_past_to_ego_curr(T_ego_first_ego_n: np.ndarray) -> np.ndarray:
    """
    计算上一帧到当前帧的相对变换矩阵。
    
    Args:
        T_ego_first_ego_n (np.ndarray): [T, 4, 4]
                                         
    Returns:
        T_ego_curr_ego_past (np.ndarray): [T, 4, 4]
                                           代表从第 t-1 帧到第 t 帧的变换矩阵。
                                           第 0 帧为单位阵。
    """
    T_ego_first_ego_curr = T_ego_first_ego_n[1:]
    T_ego_curr_ego_first = closed_form_inverse_se3(T_ego_first_ego_curr)

    T_ego_first_ego_past = T_ego_first_ego_n[:-1]

    T_ego_curr_ego_past = T_ego_curr_ego_first @ T_ego_first_ego_past

    T_ego_curr_ego_past = np.concatenate([
        np.eye(4, dtype=np.float32).reshape(1, 4, 4),
        T_ego_curr_ego_past
    ])
    
    return T_ego_curr_ego_past