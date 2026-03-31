import numpy as np
import torch
from torch import Tensor
from typing import Dict, Tuple

from dvgt.utils.geometry import closed_form_inverse_se3, to_homogeneous

def scale_align(pred, gt):
    """
    对 pred 进行缩放以对齐到 gt。
    假设 pred 和 gt 的形状都是 (N, 3) 且点是一一对应的。
    """
    # 1. 去中心化 (Centering)
    # 计算重心
    mu_pred = np.mean(pred, axis=0)
    mu_gt = np.mean(gt, axis=0)
    
    # 将点云移到原点
    pred_centered = pred - mu_pred
    gt_centered = gt - mu_gt

    # 2. 计算最优缩放因子 s (Least Squares)
    # 分子：pred 和 gt 的点积之和
    numerator = np.sum(pred_centered * gt_centered)
    # 分母：pred 自身的模长平方之和
    denominator = np.sum(pred_centered ** 2)
    
    # 避免除以 0
    if denominator == 0:
        return pred
        
    s = numerator / denominator
    
    # 3. 应用缩放
    # 注意：通常我们只缩放形状，然后把位置加回去（如果需要对齐位置，应该加 mu_gt）
    # 如果你的目的是完全对齐（包括位置），使用 mu_gt
    pred_aligned = s * pred_centered + mu_gt
    
    return pred_aligned, s

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    assert src.shape == dst.shape
    N, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    Sigma = dst_c.T @ src_c / N  # (3,3)

    U, D, Vt = np.linalg.svd(Sigma) 

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / N
        s = (D * S.diagonal()).sum() / var_src
    else:
        s = np.array(1.0)

    t = mu_dst - s * R @ mu_src

    return s.astype(np.float32), R.astype(np.float32), t.astype(np.float32)

def convert_point_in_cam_first_to_ego_first(
    point_in_cam_first: Tensor,   # [B, T, V, H, W, 3]
    ego_first_to_cam_n: Tensor,   # [B, T, V, 4, 4]
) -> Tensor:  
    ego_first_to_cam_first = ego_first_to_cam_n[:, 0, 0]    # [B, 4, 4]
    cam_first_to_ego_first = closed_form_inverse_se3(ego_first_to_cam_first)    # [B, 4, 4]
    
    cam_first_to_ego_first = cam_first_to_ego_first[:, None, None, None]    # [B, 1, 1, 1, 4, 4]
    R = cam_first_to_ego_first[..., :3, :3]
    t = cam_first_to_ego_first[..., :3, 3]

    point_in_ego_first = (point_in_cam_first @ R.transpose(-1, -2)) + t[..., None, :]
    return point_in_ego_first

def convert_point_in_ego_0_to_ray_depth_in_ego_n(
    point_in_ego_0: Tensor,   # [B, T, V, H, W, 3]
    ego_n_to_ego_0: Tensor,   # [B, T, 3, 4]
) -> Tensor:
    ego_0_to_ego_n = closed_form_inverse_se3(ego_n_to_ego_0)

    point_in_ego_n = point_in_ego_0 @ ego_0_to_ego_n[:, :, None, None, :3, :3].transpose(-1, -2) \
        + ego_0_to_ego_n[:, :, None, None, None, :3, 3]
    
    ray_depth = point_in_ego_n.norm(dim=-1, p=2)
    return ray_depth

def convert_T_cam_n_cam_0_TO_T_ego_0_ego_n(
    pred_T_cam_n_cam_0: Tensor,
    T_cam_n_world: Tensor,
    T_world_ego_n: Tensor,
) -> Tensor:
    """
    1. 这里传入的都是opencv的rdf坐标系
    2. 这里的world代表ego first，消除world坐标系，可以避免转化误差
    3. T_ego_0_cam_n @ pred_T_cam_n_cam_0 @ T_cam_0_ego_n
    
    pred_T_cam_n_cam_0: [B, T, 4, 4]
    T_cam_n_world: [B, T, 4, 4]
    T_world_ego_n: [B, T, 4, 4]

    return pred_T_ego_0_ego_n: [B, T, 4, 4]
    """
    # 求解： T_ego_0_cam_n
    T_world_cam_n = closed_form_inverse_se3(T_cam_n_world)
    T_world_ego_0 = T_world_ego_n[:, 0:1]
    T_ego_0_world = closed_form_inverse_se3(T_world_ego_0)
    T_ego_0_cam_n = T_ego_0_world @ T_world_cam_n

    # 求解： T_cam_0_ego_n
    T_cam_0_world = T_cam_n_world[:, 0:1]
    T_cam_0_ego_n = T_cam_0_world @ T_world_ego_n


    # 得到: pred_T_ego_0_ego_n
    pred_T_ego_0_ego_n = T_ego_0_cam_n @ pred_T_cam_n_cam_0 @ T_cam_0_ego_n
    return pred_T_ego_0_ego_n

def convert_pred_T_world_rdf_cam_TO_T_ego_0_ego_n(
    pred_T_world_rdf_cam: Tensor,
    T_cam_n_world_flu: Tensor,
    T_world_flu_ego_n: Tensor,
):
    """
    注意gt的world坐标系是flu，ego和cam坐标系是opencv，模型输出的world是opencv
    T_ego_0_ego_n = T_ego_0_world_flu @ T_flu_rdf @ T_world_rdf_cam @ T_cam_n_ego_n

    pred_T_world_cam: [B, T, 3, 4]
    T_cam_n_world: [B, T, 3, 4]
    T_world_ego_n: [B, T, 3, 4]

    return pred_T_ego_0_ego_n: [B, T, 3, 4]
    """
    
    # 将所有输入的 3x4 矩阵转换为 4x4 齐次矩阵
    pred_T_world_rdf_cam = to_homogeneous(pred_T_world_rdf_cam).to(torch.float64)
    T_cam_n_world_flu = to_homogeneous(T_cam_n_world_flu).to(torch.float64)
    T_world_flu_ego_n = to_homogeneous(T_world_flu_ego_n).to(torch.float64)

    # 求解： T_cam_n_ego_n
    T_cam_n_ego_n = T_cam_n_world_flu @ T_world_flu_ego_n

    # 求解：T_ego_0_world_flu
    T_ego_n_world_flu = closed_form_inverse_se3(T_world_flu_ego_n)
    T_ego_0_world_flu = T_ego_n_world_flu[:, 0:1]

    # T_flu_rdf
    T_flu_rdf = np.array([
        [ 0,  0, 1, 0],
        [-1,  0, 0, 0],
        [ 0, -1, 0, 0],
        [ 0,  0, 0, 1]  
    ], dtype=np.float64)

    T_ego_0_ego_n = T_ego_0_world_flu @ T_flu_rdf @ T_world_rdf_cam @ T_cam_n_ego_n

    return T_ego_0_ego_n[..., :3, :4].to(torch.float32)


def project_points_in_ego_first_to_depth(
    points_in_ego_first: Tensor,
    ego_first_to_world: Tensor,
    extrinsics: Tensor
) -> Tensor:
    """
    points_in_ego_first (B, T, V, H, W, 3) "第一帧自车坐标系"下的3D点。
    ego_first_to_world (B, 3, 4) or (B, 4, 4)
    extrinsics (B, T, V, 3, 4) or (B, T, V, 4, 4) world to cam。

    Returns:
        Tensor: (B, T, V, H, W)
    """
    points_in_ego_first = points_in_ego_first.to(torch.float64)
    ego_first_to_world = ego_first_to_world.to(torch.float64)
    extrinsics = extrinsics.to(torch.float64)

    B, T, V, H, W, _ = points_in_ego_first.shape
    device = points_in_ego_first.device

    # 1. points_in_ego_first -> points_in_world
    R_e1_to_w = ego_first_to_world[..., :3, :3]
    t_e1_to_w = ego_first_to_world[..., :3, 3]     # (B, 3)
    points_world = points_in_ego_first @ R_e1_to_w.transpose(-1, -2)[:, None, None, None]
    points_world = points_world + t_e1_to_w[:, None, None, None, None]

    # 2. points_in_world -> points_in_camera
    R_w_to_c = extrinsics[..., :3, :3]
    t_w_to_c = extrinsics[..., :3, 3]
    points_cam = points_world @ R_w_to_c.transpose(-1, -2)[:, :, :, None]
    points_cam = points_cam + t_w_to_c[:, :, :, None, None]

    depth = points_cam[..., 2]

    depth = torch.clamp(depth, min=1, max=150)

    return depth.to(torch.float32)


def accumulate_transform_points_and_pose_to_first_frame(T_ego_curr_ego_past, points_in_ego_n, chunk_size=50):
    """
    将序列中的所有点云从各自的当前帧坐标系转换到第一帧(全局)坐标系。

    Args:
        T_ego_curr_ego_past (Tensor): [B, T, 4, 4]
            上一帧到当前帧的变换矩阵
            第0帧不可靠，不使用。
        points_in_ego_n (Tensor): [B, T, V, H, W, 3], local point
        chunk_size (int): 每次处理多少帧，防止T过大内存溢出

    Returns:
        T_ego_first_ego_n (Tensor): [B, T, 4, 4]
        points_in_ego_first (Tensor): [B, T, V, H, W, 3]
    """
    B, T, _, _ = T_ego_curr_ego_past.shape
    device = T_ego_curr_ego_past.device

    # T_ego_first_ego_n
    T_ego_first_ego_n = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, T, 1, 1)

    T_ego_past_ego_curr = closed_form_inverse_se3(T_ego_curr_ego_past)
    current_global_pose = T_ego_first_ego_n[:, 0]   # t=0是单位阵
    for t in range(1, T):
        # 相当于: T_ego_0_ego_1
        step_transform = T_ego_past_ego_curr[:, t] 
        
        # 相当于：T_ego_0_ego_0 @ T_ego_0_ego_1
        current_global_pose = current_global_pose @ step_transform
        
        T_ego_first_ego_n[:, t] = current_global_pose

    # point in ego 0
    R_global = T_ego_first_ego_n[..., :3, :3]
    t_global = T_ego_first_ego_n[..., :3, 3]

    if T > chunk_size:
        # 过大的张量，切片处理
        points_in_ego_first = torch.zeros_like(points_in_ego_n)

        for start_idx in range(0, T, chunk_size):
            end_idx = min(start_idx + chunk_size, T)            
            pts_chunk = points_in_ego_n[:, start_idx:end_idx]
            
            R_chunk = R_global[:, start_idx:end_idx].transpose(-1, -2)[:, :, None, None]
            t_chunk = t_global[:, start_idx:end_idx, None, None, None]
            
            points_in_ego_first[:, start_idx:end_idx] = pts_chunk @ R_chunk + t_chunk
    else:
        points_in_ego_first = points_in_ego_n @ R_global.transpose(-1, -2)[:, :, None, None] + t_global[:, :, None, None, None]

    return T_ego_first_ego_n, points_in_ego_first

def transform_points_in_ego_n_to_ego_first(
    points_in_ego_n: Tensor,  # [B, T, V, H, W, 3]
    ego_n_to_ego_first: Tensor    # [B, T, 4, 4]
) -> Tensor:
    ego_n_to_ego_first = ego_n_to_ego_first[:, :, None, None]   # [B, T, 1, 1, 4, 4]
    R = ego_n_to_ego_first[..., :3, :3]
    t = ego_n_to_ego_first[..., :3, 3]
    points_in_ego_first = points_in_ego_n @ R.transpose(-1, -2) + t[..., None, :]
    return points_in_ego_first

def convert_camera_0_pose_to_ego_0_pose(
    pred_T_cam_n_cam_0: Tensor,   # [B, T, 4, 4]
    batch: Dict,
    use_umeyama_per_scene: bool = False,
) -> Tensor:
    if use_umeyama_per_scene:  # align pose translation
        device = pred_T_cam_n_cam_0.device
        B, T = pred_T_cam_n_cam_0.shape[:2]
        pred_pose_xyz_flaten = pred_T_cam_n_cam_0[..., :3, -1].reshape(-1, 3).cpu().numpy()
        gt_pose_xyz_flaten = batch['cam_first_to_cam_n'][:, :, 0, :3, -1].reshape(-1, 3).cpu().numpy()
        s, R, t = umeyama_alignment(pred_pose_xyz_flaten, gt_pose_xyz_flaten, with_scale=True)

        pred_pose_xyz_flaten = (s * (R @ pred_pose_xyz_flaten.T)).T + t
        pred_T_cam_n_cam_0[..., :3, -1] = torch.from_numpy(pred_pose_xyz_flaten.reshape(B, T, 3)).to(device)

    pred_ego_n_to_ego_first = convert_T_cam_n_cam_0_TO_T_ego_0_ego_n(
        pred_T_cam_n_cam_0=pred_T_cam_n_cam_0,     # [B, T, 4, 4]
        T_cam_n_world=batch['ego_first_to_cam_n'][:, :, 0],  # [B, T, 4, 4]
        T_world_ego_n=batch['ego_n_to_ego_first']         # [B, T, 4, 4]
    )

    return pred_ego_n_to_ego_first


def convert_camera_0_point_to_ego_0_point(
    pred_points_in_cam0: Tensor,  # [B, T, V, H, W, 3]
    batch: Dict,
    use_umeyama_per_scene: bool = False,
) -> Tuple[Tensor, Tensor]:

    if use_umeyama_per_scene:
        gt_masks = batch['point_masks']
        pred_scene_pc = pred_points_in_cam0[gt_masks].cpu().numpy()
        gt_scene_pc = batch['points_in_cam_first'][gt_masks].cpu().numpy()

        # Empty frame detected. Skipping... (Excluded from metric evaluation).
        if pred_scene_pc.shape[0] > 0 or gt_scene_pc.shape[0] > 0:
            s, R, t = umeyama_alignment(pred_scene_pc, gt_scene_pc, with_scale=True)
            pred_points_aligned_flat = (s * (R @ pred_scene_pc.T)).T + t
            
            pred_points_aligned = torch.zeros_like(pred_points_in_cam0)
            pred_points_aligned[gt_masks] = torch.from_numpy(pred_points_aligned_flat).to(pred_points_in_cam0.device)
            
            # update points
            pred_points_in_cam0 = pred_points_aligned


    pred_points_in_ego_first = convert_point_in_cam_first_to_ego_first(pred_points_in_cam0, batch['ego_first_to_cam_n'])
    
    ray_depth = convert_point_in_ego_0_to_ray_depth_in_ego_n(pred_points_in_ego_first, batch["ego_n_to_ego_first"])

    return  ray_depth, pred_points_in_ego_first
