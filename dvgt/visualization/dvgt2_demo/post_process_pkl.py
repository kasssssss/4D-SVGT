import pickle
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch # [修改/优化] 引入 torch

def closed_form_inverse_se3(se3):
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

def load_seg_data(seg_dir):
    """
    从指定的 seg_dir 读取分割数据。
    返回 shape 为 [T, V, H, W] 的 numpy array。
    """
    views = sorted([d for d in os.listdir(seg_dir) if d.startswith('view_') and os.path.isdir(os.path.join(seg_dir, d))])
    first_view_dir = os.path.join(seg_dir, views[0])
    frames = sorted([f for f in os.listdir(first_view_dir) if f.endswith('.png')])
    
    T = len(frames)
    V = len(views)
    H, W = cv2.imread(os.path.join(first_view_dir, frames[0])).shape[:2]
    seg_data = np.zeros((T, V, H, W), dtype=bool)
    
    for t_idx in range(T):
        frame_name = frames[t_idx]
        for v_idx in range(V):
            view_name = views[v_idx]
            img_path = os.path.join(seg_dir, view_name, frame_name)
            img = cv2.imread(img_path)
            seg_data[t_idx, v_idx] = (img[:, :, 0] != 255)
    return seg_data

def preprocess_point_cloud():
    data_path = 'path/to/your/saved_data.pkl'
    output_path = str(Path(data_path).parent / 'process_data_pred_wego_seg_40.pkl')

    seg_mask = load_seg_data(str(Path(data_path).parent / 'seg_per_view'))

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    window_size = 40 # 例如 0-4 帧，即滑动窗口大小为 5
    ego_window_size = 0    # 自车自身xm范围的点云的累积窗口
    ego_threshold = 12      # 自车附近20m的点云都会被累积

    args_conf_percentile = 25
    future_points_num = 0       # 加入未来帧，静态点云
    use_pose = 'pred_ego_n_to_ego_first' # 'pred_ego_n_to_ego_first', 'ego_n_to_ego_first'
    use_traj = 'pred_each_frame_best_traj_rdf' # 'pred_each_frame_best_traj_rdf', 'gt_each_frame_best_traj_rdf'
    use_points = 'pred_points' # 'pred_points', 'points'

    # 去除 B 维度 (假设 B=1)
    # images: (T, V, 3, H, W) -> 转换为 (T, V, H, W, 3) 以匹配点云维度
    images = data['images'][0].transpose(0, 1, 3, 4, 2) 
    pred_points = data[use_points][0]      # (T, V, H, W, 3)
    point_masks = data['point_masks'][0]      # (T, V, H, W)
    pred_points_conf = data['pred_points_conf'][0] # (T, V, H, W)
    pred_ray_depth = data['pred_ray_depth'][0]  # (T, V, H, W)
    poses = data[use_pose][0]                 # (T, 4, 4)
    pred_traj = data[use_traj][0] # (T, 8, 3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cutoff = np.percentile(pred_points_conf, args_conf_percentile)
    current_mask = (pred_points_conf >= cutoff) & point_masks 
    history_mask = current_mask & seg_mask  
    ego_mask = history_mask & (pred_ray_depth < ego_threshold) & (pred_points[..., 1] > 0.3)  # 地面上的点才要

    # GPU
    images_t = torch.from_numpy(images).to(device)
    pred_points_t = torch.from_numpy(pred_points.astype(np.float64)).to(device)
    poses_t = torch.from_numpy(poses.astype(np.float64)).to(device)

    current_mask_t = torch.from_numpy(current_mask).to(device)
    history_mask_t = torch.from_numpy(history_mask).to(device)
    ego_mask_t = torch.from_numpy(ego_mask).to(device)

    T = pred_points.shape[0]
    
    with open(output_path, 'wb') as f:

        for t in tqdm(range(T)):
            # [修改/优化] 重新划分不重叠的时间区间，避免点云重复累加
            curr_idx = t  # 仅当前帧
            hist_start = max(0, t - window_size + 1)
            hist_end = t  # 不包含当前帧 t
            ego_start = max(0, t - ego_window_size + 1)
            # ego_end = t+1  # debug ego 范围
            ego_end = hist_start  # 不包含近期历史 hist_start 到 t

            future_end = min(T, t + future_points_num + 1)

            pts_list = []
            col_list = []

            # 1. 提取当前帧数据
            m_curr = current_mask_t[curr_idx]
            pts_list.append(pred_points_t[curr_idx][m_curr])
            col_list.append(images_t[curr_idx][m_curr])

            # 2. 提取近期历史帧数据 (hist_start <= tau < t)
            if hist_end > hist_start:
                m_hist = history_mask_t[hist_start:hist_end]
                pts_list.append(pred_points_t[hist_start:hist_end][m_hist])
                col_list.append(images_t[hist_start:hist_end][m_hist])

            # 3. 提取远期自车盲区填补数据 (ego_start <= tau < hist_start)
            if ego_end > ego_start:
                m_ego = ego_mask_t[ego_start:ego_end]
                pts_list.append(pred_points_t[ego_start:ego_end][m_ego])
                col_list.append(images_t[ego_start:ego_end][m_ego])
            
            # 4. 提取未来帧点云
            if future_points_num > 0 and t < future_end:
                m_future = history_mask_t[t+1:future_end]
                pts_list.append(pred_points_t[t+1:future_end][m_future])
                col_list.append(images_t[t+1:future_end][m_future])
                
            valid_points = torch.cat(pts_list, dim=0)
            valid_colors = torch.cat(col_list, dim=0)
            
            # [修改/优化] 使用 GPU 进行矩阵逆运算和坐标系转换
            inv_M = closed_form_inverse_se3(poses_t[t])  # (4, 4)
            R = inv_M[:3, :3]
            trans = inv_M[:3, 3]

            # 替代齐次坐标矩阵运算 (更省内存和计算量)：P_new = P @ R^T + T
            transformed_points = torch.matmul(valid_points, R.T) + trans
            
            # 将结果拉回 CPU 并转为 NumPy 存入 Pickle
            frame_data = {
                'frame_id': t,
                'points': transformed_points.cpu().numpy().astype(np.float32),
                'colors': valid_colors.cpu().numpy().astype(np.float32), # 如果原始是 uint8，这里可能要改回 uint8 节省空间
                'traj': pred_traj[t].astype(np.float32)
            }

            pickle.dump(frame_data, f)
            del valid_points, valid_colors, transformed_points

    print(f"预处理完成，保存了 {T} 帧数据到 {output_path}")

if __name__ == "__main__":
    preprocess_point_cloud()