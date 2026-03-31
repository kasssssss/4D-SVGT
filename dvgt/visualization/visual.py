import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import numpy as np
from pathlib import Path
from time import sleep
import viser

from dvgt.visualization.utils import (
    load_pickle_data,
    center_data,
    process_and_filter_points,
    visualize_ego_poses,
    apply_sky_segmentation,
    depth_edge,            
    points_to_normals,      
    normals_edge,           
    save_point_cloud,
    save_images,
    save_depths
)

def filter_data(data, frame_indices, camera_indices):
    """
    统一根据 frame_indices 和 camera_indices 筛选 data 中的数据。
    """
    # 1. 定义哪些 Key 需要筛选什么维度
    # (T, V, ...)
    keys_tv = [
        'points', 'point_masks', 'ray_depth', 'images', 'depths',
        'pred_points', 'pred_points_conf', 'pred_ray_depth', 
    ]
    # (T, ...)
    keys_t = ['ego_n_to_ego_first', 'pred_ego_n_to_ego_first', 'ego_first_to_cam_n']
    # (V, ...)
    keys_v = ['cam_types']

    filtered_data = data # 浅拷贝，直接修改字典内容

    # 2. 执行筛选
    for key, val in filtered_data.items():
        if val is None: continue

        # 处理 (T, V) 类型
        if key in keys_tv:
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                if frame_indices:
                    val = val[frame_indices]
                if camera_indices:
                    val = val[:, camera_indices]
                filtered_data[key] = val
        
        # 处理 (T) 类型
        elif key in keys_t:
            if isinstance(val, np.ndarray) and val.shape[0] > 0:
                if frame_indices:
                    val = val[frame_indices]
                filtered_data[key] = val
        
        # 处理 (V) 类型
        elif key in keys_v:
            if camera_indices:
                if isinstance(val, list):
                    val = [val[i] for i in camera_indices]
                elif isinstance(val, np.ndarray):
                    val = val[camera_indices]
                filtered_data[key] = val
                
    return filtered_data

def compute_error_analysis(pred_points, gt_points, images, combined_mask, gt_masks, ego_poses, args):
    """
    计算误差并返回用于可视化的数据结构。
    """
    print(f"!!! Visualizing Errors > {args.error_threshold}m !!!")
    
    # 必须在有效区域内比较 (GT Mask & Pred Valid Mask)
    valid_comp_mask = combined_mask & gt_masks
    
    pred_valid = pred_points[valid_comp_mask]
    gt_valid = gt_points[valid_comp_mask]
    colors_valid = images[valid_comp_mask]
    
    # 计算 L2 误差
    errors = np.linalg.norm(pred_valid - gt_valid, axis=1)
    error_mask = errors > args.error_threshold
    
    high_error_pred = pred_valid[error_mask]
    high_error_gt = gt_valid[error_mask]
    high_error_colors = colors_valid[error_mask]
    
    print(f"Found {len(high_error_pred)} high error points.")
    
    # 偏移 GT 以便对比 (例如沿X轴偏移 50米)
    offset_val = 50.0
    high_error_gt_shifted = high_error_gt.copy()
    high_error_gt_shifted[:, 0] += offset_val
    
    # 合并 (Pred + Shifted GT)
    combined_pts_raw = np.vstack([high_error_pred, high_error_gt_shifted])
    # combined_colors = np.vstack([high_error_colors, high_error_colors])
    
    # 居中
    if args.center:
        combined_pts_raw, ego_poses, center = center_data(combined_pts_raw, ego_poses)
    
    # 拆分回两部分用于分别命名可视化
    split_idx = len(high_error_pred)
    pred_centered = combined_pts_raw[:split_idx]
    gt_centered = combined_pts_raw[split_idx:]
    
    vis_list = [
        (pred_centered, high_error_colors, "pred_error_>_thresh"),
        (gt_centered, high_error_colors, f"gt_corresponding_offset_+{offset_val}m")
    ]
    
    save_data = (combined_pts_raw, ego_poses)
    
    return vis_list, ego_poses, save_data

def visualize_gt(data, args):
    """
    处理并准备 GT 点云。
    """
    print("--- Mode: Ground Truth ---")
    
    # 1. 筛选数据
    frame_indices = args.frame_indices if args.frame_indices else list(range(data['points'].shape[0]))
    camera_indices = args.camera_indices if args.camera_indices else list(range(data['points'].shape[1]))
    data = filter_data(data, frame_indices, camera_indices)

    gt_points = data['points']   # (T, V, H, W, 3)
    gt_masks = data['point_masks']     # (T, V, H, W)
    images = data['images']         # (T, V, H, W, 3)
    ego_poses = data['ego_n_to_ego_first'] # (T, 4, 4)

    # 2. Mask 处理
    combined_mask = gt_masks.copy()
    if args.use_sky_mask:
        sky_valid = apply_sky_segmentation(images)
        combined_mask &= sky_valid

    # 3. 过滤 & 下采样
    points_final, colors_final = process_and_filter_points(
        gt_points, images, combined_mask, args.max_depth, args.downsample_ratio
    )

    # 4. 居中
    if args.center:
        points_final, ego_poses, center = center_data(points_final, ego_poses)

    return [(points_final, colors_final, "gt_point_cloud")], ego_poses, (points_final, colors_final)


def visualize_pred(data, args):
    """
    处理并准备 Pred 点云，包含误差分析功能。
    """
    print("--- Mode: Prediction ---")
    
    # 1. 筛选数据
    frame_indices = args.frame_indices if args.frame_indices else list(range(data['pred_points'].shape[0]))
    camera_indices = args.camera_indices if args.camera_indices else list(range(data['pred_points'].shape[1]))
    data = filter_data(data, frame_indices, camera_indices)

    pred_points = data['pred_points']
    ego_poses = data['pred_ego_n_to_ego_first']
    images = data['images']
    
    gt_points = data.get('points', None)
    gt_masks = data.get('point_masks', None)
    
    pred_conf = data.get('pred_points_conf', None)
    ray_depth = data.get('pred_ray_depth', None)

    # 2. 构建 Mask
    T, V, H, W, _ = pred_points.shape
    combined_mask = np.ones((T, V, H, W), dtype=bool)

    if args.use_gt_mask and gt_masks is not None:
        print("Applying GT Mask...")
        combined_mask &= gt_masks

    if args.conf_percentile > 0 and pred_conf is not None:
        print(f"Filtering lowest {args.conf_percentile}% confidence...")
        cutoff = np.percentile(pred_conf, args.conf_percentile)
        combined_mask &= (pred_conf >= cutoff)

    if args.use_sky_mask:
        sky_valid = apply_sky_segmentation(images)
        combined_mask &= sky_valid
    
    if args.use_edge_masks:
        print("Applying Edge Masks...")
        for t in range(T):
            for v in range(V):
                frame_pts = pred_points[t, v]
                # 如果有 ray_depth 用 ray_depth，否则算模长
                frame_d = ray_depth[t, v] if ray_depth is not None else np.linalg.norm(frame_pts, axis=-1)
                
                # Depth Edge
                d_edge = depth_edge(frame_d, rtol=args.edge_depth_rtol, mask=combined_mask[t,v])
                # Normal Edge
                norm, n_mask = points_to_normals(frame_pts, mask=combined_mask[t,v])
                n_edge = normals_edge(norm, tol=args.edge_normal_tol, mask=n_mask)
                
                combined_mask[t, v] &= ~(d_edge | n_edge)

    # 3. 误差分析 (如果启用)
    if args.error_threshold > 0 and gt_points is not None:
        return compute_error_analysis(pred_points, gt_points, images, combined_mask, gt_masks, ego_poses, args)

    # 4. 常规流程：过滤 & 下采样
    points_final, colors_final = process_and_filter_points(
        pred_points, images, combined_mask, args.max_depth, args.downsample_ratio
    )
    
    # 5. 居中
    if args.center:
        points_final, ego_poses, center = center_data(points_final, ego_poses)
    
    return [(points_final, colors_final, "pred_point_cloud")], ego_poses, (points_final, colors_final)

def main():
    parser = argparse.ArgumentParser(description="DVGT Point Cloud Visualizer & Analyzer")
    
    # 核心路径与模式
    parser.add_argument('--pkl_path', type=str, required=True, help="Path to saved_data.pkl")
    parser.add_argument('--mode', type=str, default='pred', choices=['gt', 'pred'], help="Visualize 'gt' or 'pred'")
    parser.add_argument('--save_info', action='store_true', help="If set, saves .ply and images to this folder")
    
    # 筛选与Mask参数
    parser.add_argument('--frame_indices', default=list(range(350)), type=int, nargs='+', help="List of frame indices to visualize. Default: all.")
    parser.add_argument('--camera_indices', default=[], type=int, nargs='+', help="List of camera indices to visualize. Default: all.")
    parser.add_argument('--downsample_ratio', type=float, default=0.1, help="0.0-1.0 to downsample")
    parser.add_argument('--max_depth', type=float, default=-1, help="Max depth cut-off")
    parser.add_argument('--center', action='store_true', help="是否将点云和pose中心化")

    # Pred 特有参数
    parser.add_argument('--use_gt_mask', action='store_true', help="[Pred] Filter pred using GT mask")
    parser.add_argument('--conf_percentile', type=float, default=0, help="[Pred] Filter bottom X% confidence")
    parser.add_argument('--error_threshold', type=float, default=-1, help="[Pred] Show points with error > threshold")
    
    # 视觉参数
    parser.add_argument('--use_sky_mask', action='store_true', help="Remove sky")
    parser.add_argument('--use_edge_masks', action='store_true', help="Remove depth/normal edges")
    parser.add_argument('--edge_depth_rtol', type=float, default=0.1)
    parser.add_argument('--edge_normal_tol', type=float, default=50)
    
    # Viser 参数
    parser.add_argument('--no_viser', action='store_true', help="Skip interactive viz, just save")
    parser.add_argument('--no_ego', action='store_true', help="Hide ego poses")
    parser.add_argument('--no_point', action='store_true', help="Hide ego poses")
    parser.add_argument('--point_size', type=float, default=0.01)

    args = parser.parse_args()

    # 1. 加载数据
    data = load_pickle_data(args.pkl_path)
    
    # 2. 根据模式处理数据
    if args.mode == 'gt':
        vis_data, poses, save_data_tuple = visualize_gt(data, args)
    elif args.mode == 'pred':
        vis_data, poses, save_data_tuple = visualize_pred(data, args)
    
    # 3. 保存文件 (如果指定)
    if args.save_info:
        save_path = Path(args.pkl_path).parent / "save_info"
        print(f"Saving outputs to {save_path}...")
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 解包 save_data_tuple
        points_to_save, colors_to_save = save_data_tuple
        
        save_point_cloud(points_to_save, colors_to_save, save_path, save_type='glb')
        
        # 兼容 cam_types 不存在的情况
        cam_types = data.get('cam_types', [f'cam_{i}' for i in range(10)])

        # 保存辅助数据 (RGB/Depth)
        if 'images' in data:
            save_images(data['images'], save_path / 'rgb', cam_types)
        if 'depths' in data:
            save_depths(data['depths'], save_path / 'cam_depth', cam_types)
        if args.mode == 'gt' and 'ray_depth' in data:
            save_depths(data['ray_depth'], save_path / 'gt_ray_depth', cam_types)
        if args.mode == 'pred' and 'pred_ray_depth' in data:
            save_depths(data['pred_ray_depth'], save_path / 'pred_ray_depth', cam_types)

    # 4. 启动 Viser
    if not args.no_viser:
        server = viser.ViserServer()
        print("\n--- Starting Viser Server ---")

        # 控制viser开始的位置
        scene_center = np.array([0, 0, 0])
        if len(vis_data) > 0 and len(vis_data[0][0]) > 0:
            # vis_data[0][0] 是第0帧点云
            scene_center = np.mean(vis_data[0][0], axis=0)

        start_pos = np.array([-5, 0, 5]) # 默认偏移
        if len(poses) > 0:
             # poses 是 4x4 矩阵，取平移部分 (前三行第四列)
             # 让相机站在第一个轨迹点后面一点
             start_pos = poses[0][:3, 3] + np.array([-5.0, 0.0, 2.0])

        @server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.look_at = scene_center
            client.camera.position = start_pos

        # GUI: 显示当前位置
        @server.on_client_connect
        def _(client: viser.ClientHandle):
            print(f"Client {client.client_id} connected.")
            with client.gui.add_folder("Camera Info"):
                gui_cam_wxyz = client.gui.add_text("Cam WXYZ (quat)", initial_value="...")
                gui_cam_pos = client.gui.add_text("Cam Position (xyz)", initial_value="...")

            @client.camera.on_update
            def _(camera: viser.CameraHandle):
                gui_cam_wxyz.value = str(np.round(camera.wxyz, 3))
                gui_cam_pos.value = str(np.round(camera.position, 3))
            
            _(client.camera)

        # GUI: 控制点大小
        with server.gui.add_folder("Controls"):
            point_size_slider = server.gui.add_slider(
                "Point Size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size
            )

        if not args.no_point:
            pc_handles = []
            for points, colors, name in vis_data:
                # 打印点云信息
                print(f"[{name}] Shape: {points.shape}, Mean Center: {np.mean(points, axis=0)}")

                if len(points) == 0: continue
                handle = server.scene.add_point_cloud(
                    name=f"/{name}",
                    points=points,
                    colors=colors,
                    point_size=point_size_slider.value,
                )
                pc_handles.append(handle)
                
            @point_size_slider.on_update
            def _(_):
                for h in pc_handles: h.point_size = point_size_slider.value

        if not args.no_ego:
            visualize_ego_poses(server, poses)
        
        print("Viser running. Press Ctrl+C to stop.")
        while True:
            sleep(1)

if __name__ == "__main__":
    main()