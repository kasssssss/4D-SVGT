from typing import Dict, List
import torch
from torch import Tensor
import numpy as np

def compute_trajectory_errors(pred_trajectory: Tensor, gt_trajectory: Tensor) -> Dict[str, List[float]]:
    """
    评测逐秒的轨迹误差

    Args:
        pred_trajectory: [B, N, 3 or 4] (x, y, theta) or (x, y, z, theta)
        gt_trajectory:   [B, N, 3 or 4] (x, y, theta) or (x, y, z, theta)

    Returns:
        Dict: 
            key: "1s_xy", "1s_theta", "2s_xy"...
            value: List[np.ndarray] 长度为 BatchSize，每个元素是该样本在该时刻的误差值。
    """
    mask = (pred_trajectory[..., 0] == -999.0)
    
    # 拆分 XY 和 Theta
    pred_xy = pred_trajectory[..., :-1]
    gt_xy = gt_trajectory[..., :-1]

    pred_theta = pred_trajectory[..., -1]
    gt_theta = gt_trajectory[..., -1]

    # 3. 计算 XY 的 L2 Error (Euclidean Distance)
    xy_errors = torch.norm(pred_xy - gt_xy, p=2, dim=-1)

    # 4. 计算 Theta 的 Wrapped L1 Error
    # 公式: (diff + pi) % 2pi - pi -> 绝对值
    diff = pred_theta - gt_theta
    diff_wrapped = (diff + torch.pi) % (2 * torch.pi) - torch.pi
    theta_errors = torch.abs(diff_wrapped) # [B, T_key]

    metrics_dict = {}
    num_timestamps = xy_errors.shape[1] # T_key

    # mask 无效数据
    xy_errors[mask] = torch.nan
    theta_errors[mask] = torch.nan

    xy_errors_np = xy_errors.detach().cpu().numpy()
    theta_errors_np = theta_errors.detach().cpu().numpy()

    for t in range(num_timestamps):
        metrics_dict[f"{t}_xy_l2"] = xy_errors_np[:, t].tolist()
        metrics_dict[f"{t}_theta_l1"] = theta_errors_np[:, t].tolist()

    return metrics_dict