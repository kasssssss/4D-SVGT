"""
    单帧，单图是torch版本快
    单帧，多图是np版本快
    多帧，多图是np版本快
"""
from collections import defaultdict
import re
import torch
from kaolin.metrics.pointcloud import sided_distance
import numpy as np
from scipy.spatial import cKDTree

def compute_chamfer_distance_multi_view(pred_points: torch.Tensor, gt_points: torch.Tensor, gt_masks: torch.Tensor):
    """
        因为自驾是动态场景，这里计算单帧多视角的chamfer
        pred_points: [B, T, V, H, W, 3]
        gt_points: [B, T, V, H, W, 3]
        gt_masks: [B, T, V, H, W]
    """
    B, T, V, H, W, _ = pred_points.shape
    
    num_images = B * T
    pred_points = pred_points.reshape(num_images, V * H * W, 3).contiguous()
    gt_points = gt_points.reshape(num_images, V * H * W, 3).contiguous()
    gt_masks = gt_masks.reshape(num_images, V * H * W).contiguous()

    accuracies = []
    completenesses = []
    chamfers = []

    for i in range(num_images):
        mask = gt_masks[i]
        valid_pred_pc = pred_points[i][mask]
        valid_gt_pc = gt_points[i][mask]

        if valid_pred_pc.shape[0] == 0 or valid_gt_pc.shape[0] == 0:
            continue

        # 匹配API要求
        valid_pred_pc = valid_pred_pc.unsqueeze(0)  # Shape: [1, N, 3]
        valid_gt_pc = valid_gt_pc.unsqueeze(0)      # Shape: [1, M, 3]

        # Accuracy: 从预测点云到真实点云的距离
        # 表示预测出的点离真实的物体表面有多近。如果值很小，说明你预测的点都很好地落在了真实物体的表面上。
        # kaolin 返回的是squared euclidean distances，所以需要开方
        dist_p_to_g_sq, _ = sided_distance(valid_pred_pc, valid_gt_pc)
        accuracy = torch.sqrt(dist_p_to_g_sq).mean()

        # Completeness: 从真实点云到预测点云的距离
        # 真实物体的表面是否被预测点云全面覆盖了。如果值很小，说明真实物体的每个部分附近都有预测点存在，预测得比较完整。
        dist_g_to_p_sq, _ = sided_distance(valid_gt_pc, valid_pred_pc)
        completeness = torch.sqrt(dist_g_to_p_sq).mean()
        
        chamfer = accuracy + completeness

        accuracies.append(accuracy.item())
        completenesses.append(completeness.item())
        chamfers.append(chamfer.item())

    if not accuracies:
        return {'accuracy': 0.0, 'completeness': 0.0, 'chamfer': 0.0}

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_completeness = sum(completenesses) / len(completenesses)
    avg_chamfer = sum(chamfers) / len(chamfers)

    return {
        'accuracy': avg_accuracy,
        'completeness': avg_completeness,
        'chamfer': avg_chamfer
    }

def compute_chamfer_distance_multi_view_np(pred_points: np.ndarray, gt_points: np.ndarray, gt_masks: np.ndarray):
    """
        因为自驾是动态场景，这里计算单帧多视角的chamfer
        pred_points: [B, T, V, H, W, 3]
        gt_points: [B, T, V, H, W, 3]
        gt_masks: [B, T, V, H, W]
    """
    B, T, V, H, W, _ = pred_points.shape
    
    num_images = B * T
    pred_points = pred_points.reshape(num_images, V * H * W, 3).cpu().numpy()
    gt_points = gt_points.reshape(num_images, V * H * W, 3).cpu().numpy()
    gt_masks = gt_masks.reshape(num_images, V * H * W).cpu().numpy()

    accuracies = []
    completenesses = []
    chamfers = []

    for i in range(num_images):

        mask = gt_masks[i]
        valid_pred_pc = pred_points[i][mask]
        valid_gt_pc = gt_points[i][mask]

        if valid_pred_pc.shape[0] == 0 or valid_gt_pc.shape[0] == 0:
            continue

        # Accuracy: 从预测到GT的最近距离
        gt_tree = cKDTree(valid_gt_pc)
        dist_p_to_g, _ = gt_tree.query(valid_pred_pc, k=1)
        accuracy = np.mean(dist_p_to_g)

        # Completeness: 从GT到预测的最近距离
        pred_tree = cKDTree(valid_pred_pc)
        dist_g_to_p, _ = pred_tree.query(valid_gt_pc, k=1)
        completeness = np.mean(dist_g_to_p)
        
        # Chamfer
        chamfer = accuracy + completeness

        accuracies.append(accuracy)
        completenesses.append(completeness)
        chamfers.append(chamfer)

    if not accuracies:
        return {'accuracy': 0.0, 'completeness': 0.0, 'chamfer': 0.0}

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_completeness = sum(completenesses) / len(completenesses)
    avg_chamfer = sum(chamfers) / len(chamfers)

    return {
        'accuracy': avg_accuracy,
        'completeness': avg_completeness,
        'chamfer': avg_chamfer
    }

def compute_chamfer_distance_single_view(pred_points: torch.Tensor, gt_points: torch.Tensor, gt_masks: torch.Tensor, prefix: str = ""):
    """
        因为自驾是动态场景，这里计算单帧单视角的chamfer
        pred_points: [B, T, V, H, W, 3]
        gt_points: [B, T, V, H, W, 3]
        gt_masks: [B, T, V, H, W]
    """
    B, T, V, H, W, _ = pred_points.shape
    
    num_images = T * V
    pred_points = pred_points.reshape(B, num_images, H * W, 3).contiguous()
    gt_points = gt_points.reshape(B, num_images, H * W, 3).contiguous()
    gt_masks = gt_masks.reshape(B, num_images, H * W).contiguous()

    result = defaultdict(list)

    for b in range(B):
        accuracies = []
        completenesses = []
        chamfers = []
        for i in range(num_images):
            mask = gt_masks[b, i]
            if not mask.any(): # 如果mask全为0，不计算指标
                continue

            valid_pred_pc = pred_points[b, i][mask]
            valid_gt_pc = gt_points[b, i][mask]

            if valid_pred_pc.shape[0] == 0 or valid_gt_pc.shape[0] == 0:
                continue
            
            # 匹配API要求
            valid_pred_pc = valid_pred_pc.unsqueeze(0)  # Shape: [1, N, 3]
            valid_gt_pc = valid_gt_pc.unsqueeze(0)      # Shape: [1, M, 3]

            # Accuracy: 从预测点云到真实点云的距离
            # 表示预测出的点离真实的物体表面有多近。如果值很小，说明你预测的点都很好地落在了真实物体的表面上。
            # kaolin 返回的是squared euclidean distances，所以需要开方
            dist_p_to_g_sq, _ = sided_distance(valid_pred_pc, valid_gt_pc)
            accuracy = torch.sqrt(dist_p_to_g_sq).mean()

            # Completeness: 从真实点云到预测点云的距离
            # 真实物体的表面是否被预测点云全面覆盖了。如果值很小，说明真实物体的每个部分附近都有预测点存在，预测得比较完整。
            dist_g_to_p_sq, _ = sided_distance(valid_gt_pc, valid_pred_pc)
            completeness = torch.sqrt(dist_g_to_p_sq).mean()
            
            chamfer = accuracy + completeness

            accuracies.append(accuracy.item())
            completenesses.append(completeness.item())
            chamfers.append(chamfer.item())
        
        if not accuracies:      # 说明这一组的mask全部为0，直接跳过，不计算指标
            result[f'{prefix}accuracy'].append(np.nan)
            result[f'{prefix}completeness'].append(np.nan)
            result[f'{prefix}chamfer'].append(np.nan)
        else:
            result[f'{prefix}accuracy'].append(np.mean(accuracies))
            result[f'{prefix}completeness'].append(np.mean(completenesses))
            result[f'{prefix}chamfer'].append(np.mean(chamfers))

    return result

def compute_chamfer_distance_single_view_np(pred_points: np.ndarray, gt_points: np.ndarray, gt_masks: np.ndarray):
    """
        因为自驾是动态场景，这里计算单帧单视角的chamfer
        pred_points: [B, T, V, H, W, 3]
        gt_points: [B, T, V, H, W, 3]
        gt_masks: [B, T, V, H, W]
    """
    B, T, V, H, W, _ = pred_points.shape
    
    num_images = B * T * V
    pred_points = pred_points.reshape(num_images, H * W, 3).cpu().numpy()
    gt_points = gt_points.reshape(num_images, H * W, 3).cpu().numpy()
    gt_masks = gt_masks.reshape(num_images, H * W).cpu().numpy()

    accuracies = []
    completenesses = []
    chamfers = []

    for i in range(num_images):

        mask = gt_masks[i]
        valid_pred_pc = pred_points[i][mask]
        valid_gt_pc = gt_points[i][mask]

        if valid_pred_pc.shape[0] == 0 or valid_gt_pc.shape[0] == 0:
            continue

        # Accuracy: 从预测到GT的最近距离
        gt_tree = cKDTree(valid_gt_pc)
        dist_p_to_g, _ = gt_tree.query(valid_pred_pc, k=1)
        accuracy = np.mean(dist_p_to_g)

        # Completeness: 从GT到预测的最近距离
        pred_tree = cKDTree(valid_pred_pc)
        dist_g_to_p, _ = pred_tree.query(valid_gt_pc, k=1)
        completeness = np.mean(dist_g_to_p)
        
        # Chamfer
        chamfer = accuracy + completeness

        accuracies.append(accuracy)
        completenesses.append(completeness)
        chamfers.append(chamfer)

    if not accuracies:
        return {'accuracy': 0.0, 'completeness': 0.0, 'chamfer': 0.0}

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_completeness = sum(completenesses) / len(completenesses)
    avg_chamfer = sum(chamfers) / len(chamfers)

    return {
        'accuracy': avg_accuracy,
        'completeness': avg_completeness,
        'chamfer': avg_chamfer
    }