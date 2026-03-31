import torch
from torch import Tensor
import numpy as np

def depth_evaluation(pred_depth: Tensor, gt_depth: Tensor, gt_mask: Tensor, prefix: str = ''):
    """
    计算 Abs Rel 和 δ < 1.25 指标。
    pred_depth: B, T, V, H, W
    gt_depth: B, T, V, H, W
    gt_mask: B, T, V, H, W 
    """
    gt_mask = gt_mask.bool()

    result = {f'{prefix}Abs Rel': [], f'{prefix}δ < 1.25': []}
    
    B = pred_depth.shape[0]
    min_depth = 1e-6 

    for b in range(B):
        cur_mask = gt_mask[b]
        if not cur_mask.any():  # 无效帧不计算指标
            result[f'{prefix}Abs Rel'].append(np.nan)
            result[f'{prefix}δ < 1.25'].append(np.nan)
            continue

        pred = pred_depth[b][cur_mask]
        gt = gt_depth[b][cur_mask]

        pred = pred.clamp(min=min_depth)
        gt = gt.clamp(min=min_depth)

        
        # 1. Abs Rel: |pred - gt| / gt
        abs_rel = (torch.abs(pred - gt) / gt).mean()

        # 2. δ < 1.25: max(pred/gt, gt/pred) < 1.25
        r1 = pred / gt
        r2 = gt / pred
        max_ratio = torch.maximum(r1, r2)
        delta_125 = (max_ratio < 1.25).float().mean()

        # 转为 Python float 存入列表 (如果需要 tensor 返回，去掉 .item())
        result[f'{prefix}Abs Rel'].append(abs_rel.item())
        result[f'{prefix}δ < 1.25'].append(delta_125.item())

    return result

def depth_evaluation_numpy(pred_depth: np.ndarray, proj_depth: np.ndarray, valid_mask: np.ndarray, prefix: str = '') -> dict:
    """
    计算 Abs Rel 和 δ < 1.25 指标 (NumPy 版本)。
    pred_depth: np.ndarray, shape (H, W)
    proj_depth: np.ndarray, shape (H, W)
    valid_mask: np.ndarray, shape (H, W)
    """
    valid_mask = valid_mask.astype(bool)

    if not valid_mask.any():
        return {
            f'{prefix}Abs Rel': np.nan, 
            f'{prefix}δ < 1.25': np.nan
        }

    pred = pred_depth[valid_mask]
    gt = proj_depth[valid_mask]

    min_depth = 1e-6
    pred = np.clip(pred, a_min=min_depth, a_max=None)
    gt = np.clip(gt, a_min=min_depth, a_max=None)

    # 1. Abs Rel: |pred - gt| / gt
    abs_rel = np.mean(np.abs(pred - gt) / gt)

    # 2. δ < 1.25: max(pred/gt, gt/pred) < 1.25
    r1 = pred / gt
    r2 = gt / pred
    max_ratio = np.maximum(r1, r2)
    delta_125 = np.mean(max_ratio < 1.25)

    return {
        f'{prefix}Abs Rel': float(abs_rel),
        f'{prefix}δ < 1.25': float(delta_125)
    }