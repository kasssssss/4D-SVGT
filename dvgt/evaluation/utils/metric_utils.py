from collections import defaultdict
import pandas as pd
import numpy as np
import logging
import torch.distributed as dist
import os

from dvgt.evaluation.metrics.point_cloud_metrics import compute_chamfer_distance_single_view
from dvgt.evaluation.metrics.depth_metrics import depth_evaluation
from dvgt.evaluation.metrics.auc_pose_metrics import calculate_pose_auc
from dvgt.evaluation.metrics.trajectory_metrics import compute_trajectory_errors

def compute_all_metrics(batch, predictions, enable_trajectory_save: bool = False):
    metrics = {}
    metrics['seq_name'] = batch['seq_name']

    if 'points_in_ego_first' in predictions:
        pc_metrics = compute_chamfer_distance_single_view(
            predictions['points_in_ego_first'], batch['points_in_ego_first'], batch['point_masks'], prefix="ego_first_"
        )
        metrics.update(pc_metrics)
    
    if 'ray_depth' in predictions:
        ray_depth_metrics = depth_evaluation(predictions['ray_depth'], batch['ray_depth'], batch['point_masks'], prefix="ego_n_")
        metrics.update(ray_depth_metrics)

    if 'ego_n_to_ego_first' in predictions:
        pose_metrics = calculate_pose_auc(
            predictions['ego_n_to_ego_first'], batch['ego_n_to_ego_first'], batch['point_masks'], prefix='ego_n_to_ego_first_'
        )
        metrics.update(pose_metrics)
        
    if 'trajectories' in predictions:
        trajectory_metrics = compute_trajectory_errors(predictions['trajectories'], batch['trajectories'])
        metrics.update(trajectory_metrics)
        if enable_trajectory_save:
            # Storing trajectories: Total memory usage for 100k scenes is ~6.8MB.
            # Safe for cross-process gathering without memory bottlenecks.
            metrics.update(
                pred_traj=[traj.flatten() for traj in predictions['trajectories'].cpu().numpy()],
                gt_traj=[traj.flatten() for traj in batch['trajectories'].cpu().numpy()],
            )
    return metrics
        
def summarize_and_log(rank, world_size, log_dir, all_metrics):
    # convert to numpy
    local_data_packed = {}
    for k, v in all_metrics.items():
        if len(v) == 0:
            continue
        if isinstance(v[0], str):
            local_data_packed[k] = v
        else:
            local_data_packed[k] = np.array(v).astype(np.float32)

    # gather all metrics
    gathered_metrics = [None] * world_size if rank == 0 else None
    dist.gather_object(local_data_packed, gathered_metrics, dst=0)

    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        temp_storage = defaultdict(list)
        for proc_metrics in gathered_metrics:
            for key, val_list in proc_metrics.items():
                temp_storage[key].append(val_list)

        final_data_flat = {}
        for k, v_list in temp_storage.items():
            if isinstance(v_list[0], list): 
                # string add: ['a1', 'c1'] + ['b1'] = ['a1', 'c1', 'b1']
                final_data_flat[k] = sum(v_list, [])
            else:
                final_data_flat[k] = np.concatenate(v_list).tolist()

        df = pd.DataFrame(final_data_flat)
        
        # duplicate removal
        logging.info(f"去重前元数据条数: {len(df)}")
        df = df.drop_duplicates(subset=['seq_name'], keep='first')
        logging.info(f"去重后元数据条数: {len(df)}")

        raw_csv_path = os.path.join(log_dir, "metrics_results.csv")
        df.to_csv(raw_csv_path, index=False)
        logging.info(f"Raw metrics saved to {raw_csv_path}")

        # calculate summary metrics
        dataset_groups = ['kitti', 'nuscene', 'waymo', 'openscene', 'ddad']
        TARGET_COLUMNS = [
            '1_xy_l2', '1_theta_l1', '3_xy_l2', '3_theta_l1', '5_xy_l2', '5_theta_l1', '7_xy_l2', '7_theta_l1',
            'ego_first_accuracy', 'ego_first_completeness', 'ego_first_chamfer', 'ego_n_Abs Rel', 'ego_n_δ < 1.25', 'ego_n_to_ego_first_AUC@30', 
            'placeholder'
        ]

        summary_values = []
        summary_headers = []

        for ds_key in dataset_groups:
            mask = df['seq_name'].str.contains(ds_key)
            group_df = df[mask]
            
            for col in TARGET_COLUMNS:
                # 构建表头，例如: nuscene_accuracy
                header_name = f"{ds_key}_{col}"
                summary_headers.append(header_name)

                # 检查列是否存在且该组是否有数据
                if not group_df.empty and col in group_df.columns:
                    # 强制转换为数字，无法转换的变为NaN
                    numeric_series = pd.to_numeric(group_df[col], errors='coerce')
                    # 计算均值，忽略NaN
                    val = np.nanmean(numeric_series)              
                    summary_values.append(val)
                else:
                    # 如果key不存在或者该组没数据，放置一个空 (这里用 None/NaN)
                    summary_values.append(None)

        summary_txt_path = os.path.join(log_dir, "summary_metrics.txt")
        summary_df = pd.DataFrame([summary_values], columns=summary_headers)
        summary_df.to_csv(summary_txt_path, sep='\t', index=False, float_format='%.3f')
        logging.info(f"Summary metrics saved to {summary_txt_path}")

        # 将结果打印到日志中
        logging.info("Evaluation Summary Table:")
        transposed_df = summary_df.T
        transposed_df.columns = ['Score']
        logging.info("\n" + transposed_df.to_string(float_format=lambda x: "{:.4f}".format(x)))