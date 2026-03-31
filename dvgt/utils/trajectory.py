import torch
from torch import Tensor

def convert_pose_rdf_to_trajectory_flu(T_ego_current_rdf_ego_future_rdf: Tensor):
    """
    将相对位姿 (RDF格式) 转换为 NAVSIM 轨迹格式 (FLU格式 [x, y, theta])。
    Yaw (Theta) = torch.atan2(R[..., 1, 0], R[..., 0, 0])，范围是[-pi, pi]
    
    Args:
        T_ego_current_rdf_ego_future_rdf (torch.Tensor): (B, N, 4, 4) 
            
    Returns:
        torch.Tensor: (B, N, 3) 
            NAVSIM 轨迹: [x, y, theta] (在 FLU 坐标系下)
    """
    device = T_ego_current_rdf_ego_future_rdf.device
    dtype = T_ego_current_rdf_ego_future_rdf.dtype

    T_flu_rdf = torch.tensor([
        [0,  0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0,  0, 0, 1]
    ], device=device, dtype=dtype).view(1, 1, 4, 4)

    # 旋转矩阵都是正交阵：p_inv = P.T
    # 这里平移是0，所以也是正交的：每列模长是1，列向量，两两点积为0
    T_rdf_flu = T_flu_rdf.transpose(-1, -2)

    T_ego_current_flu_ego_future_flu = T_flu_rdf @ T_ego_current_rdf_ego_future_rdf @ T_rdf_flu
    
    x_flu = T_ego_current_flu_ego_future_flu[..., 0, 3]
    y_flu = T_ego_current_flu_ego_future_flu[..., 1, 3]

    # atan(x)，范围是(-pi/2, pi/2)，atan2(y, x)，能区分正负号，范围是(-pi, pi)
    # 这里pose是flu，theta是yaw，也就是自车绕世界坐标系z轴旋转的角度，这个角度恰好就是自车的x轴，相比于世界坐标系x轴的旋转，也就是旋转矩阵的第0列
    # 直接取这一列的y / x计算atan2即可
    theta_flu = torch.atan2(
        T_ego_current_flu_ego_future_flu[..., 1, 0], 
        T_ego_current_flu_ego_future_flu[..., 0, 0]
    )

    trajectory = torch.stack([x_flu, y_flu, theta_flu], dim=-1)
    
    return trajectory

def convert_pose_rdf_to_trajectory_rdf(pose_rdf: torch.Tensor):
    """
    从 RDF Pose 中提取 RDF 格式的轨迹 (Right, Forward, Theta_RDF)。
    保持全程在 RDF 坐标系下，便于模型监督。
    
    Args:
        pose_rdf: (..., 4, 4) RDF 相对位姿
    
    Returns:
        traj_rdf: (..., 3) -> [right, forward, theta_rdf]
    """
    # 1. 提取位移 (Translation)
    # RDF中: x是Right(0), y是Down(1), z是Forward(2)
    # 我们只需要地平面分量: Right 和 Forward
    right_rdf = pose_rdf[..., 0, 3]   # x
    forward_rdf = pose_rdf[..., 2, 3] # z
    
    # 2. 提取角度 (Theta / Yaw)
    # 取旋转矩阵的第三列 (索引2)，这是车辆自身的Forward轴在参考系中的表示
    # R[0, 2] 是 Forward向量在 x(Right) 轴上的投影
    # R[2, 2] 是 Forward向量在 z(Forward) 轴上的投影
    theta_rdf = torch.atan2(
        pose_rdf[..., 0, 2], 
        pose_rdf[..., 2, 2]
    )
    
    traj_rdf = torch.stack([right_rdf, forward_rdf, theta_rdf], dim=-1)
    return traj_rdf

def convert_rdf_traj_to_flu_traj(traj_rdf: torch.Tensor):
    """
    将 RDF 轨迹 [right, forward, theta_rdf] 转换为 NAVSIM/FLU 轨迹 [x, y, theta_flu]
    这是纯几何变换，不需要复杂的矩阵乘法。
    
    Args:
        traj_rdf: (..., 3) [right, forward, theta_rdf]
        
    Returns:
        traj_flu: (..., 3) [x_flu, y_flu, theta_flu]
    """
    right_rdf = traj_rdf[..., 0]
    forward_rdf = traj_rdf[..., 1]
    theta_rdf = traj_rdf[..., 2]
    
    # --- 坐标映射推导 ---
    # FLU x (Forward) = RDF z (Forward)
    # FLU y (Left)    = -RDF x (Right)
    
    x_flu = forward_rdf
    y_flu = -right_rdf
    
    # --- 角度映射推导 ---
    # FLU 定义: atan2(y, x) = atan2(Left, Forward)
    # RDF 定义: atan2(x, z) = atan2(Right, Forward) = atan2(-Left, Forward)
    # 因为 atan2(-y, x) = -atan2(y, x)
    # 所以 theta_flu = -theta_rdf
    
    theta_flu = -theta_rdf
    
    return torch.stack([x_flu, y_flu, theta_flu], dim=-1)