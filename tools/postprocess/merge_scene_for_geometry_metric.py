from typing import Dict, List
from collections import defaultdict
import pandas as pd
import os
import numpy as np
from pathlib import Path
import argparse

def merge_split_scene_metrics(metrics: Dict[str, List], TARGET_COLUMNS: List[str]) -> Dict[str, List]:
    """
    根据每个clip的帧数占比，将切分场景的指标进行加权合并。

    采用两阶段方法：
    1. 遍历所有数据，将属于同一个原始场景的片段分组，并计算出总帧数。
    2. 对每个分组，用每个片段的指标乘以其归一化权重(片段帧数 / 总帧数)，然后求和。

    Args:
        metrics (Dict[str, list]): 
            从所有GPU收集并整理后的指标字典。
            例如: {'seq_name': ['A#1#10', 'A#2#2', 'B'], 'chamfer': [0.5, 0.8, 0.6]}

    Returns:
        Dict[str, list]:
            合并后的指标字典。
            例如: {'seq_name': ['B', 'A'], 'chamfer': [0.6, 0.55]}
            其中 A 的 chamfer = 0.5 * (10/12) + 0.8 * (2/12) = 0.55
    """
    # --- 阶段 1: 分组并计算每个原始场景的总长度 ---
    
    # 存储未切分场景的指标
    final_metrics = defaultdict(list) 
    
    # 存储切分场景的信息，用于后续处理
    # 结构: { original_id: [ {'seq_name': str, 'chamfer': float, ...}, ... ] }
    split_scenes_grouped = defaultdict(list)
    
    for i in range(len(metrics['seq_name'])):
        seq_name = metrics['seq_name'][i]
        original_id = seq_name.split('#')[1]
        # 将该片段的所有指标打包成一个字典，存入分组
        clip_data = {key: val_list[i] for key, val_list in metrics.items()}
        split_scenes_grouped[original_id].append(clip_data)

    # --- 阶段 2: 对每个分组进行加权求和 ---
    
    for original_id, clips_data in split_scenes_grouped.items():
        # 计算该场景所有片段的总长度
        total_length = sum(int(clip['seq_name'].split('#')[-1]) for clip in clips_data)
        
        # 初始化用于累加加权指标的字典
        weighted_sums = defaultdict(float)
        
        if total_length == 0:
            continue # 避免除以零

        # 遍历该场景的每个片段，计算加权值
        for clip in clips_data:
            clip_length = int(clip['seq_name'].split('#')[-1])
            # 计算归一化权重
            weight = clip_length / total_length
            
            # 累加每个指标的加权值
            for key, value in clip.items():
                if key in TARGET_COLUMNS:
                    weighted_sums[key] += value * weight
        
        # 将计算好的原始场景指标添加到最终结果中
        final_metrics['seq_name'].append(original_id)
        for key, total_val in weighted_sums.items():
            final_metrics[key].append(total_val)
            
    return dict(final_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    args = parser.parse_args()

    log_dir = Path(args.csv_path).parent

    df = pd.read_csv(args.csv_path)
    df = df.drop_duplicates(subset=['seq_name'], keep='first')

    # 计算每个数据集的指标
    dataset_groups = ['kitti', 'nuscene', 'waymo', 'openscene', 'ddad']
    TARGET_COLUMNS = [
        'ego_first_accuracy', 'ego_first_completeness', 'ego_first_chamfer', 
        'ego_n_Abs Rel', 'ego_n_δ < 1.25', 'ego_n_to_ego_first_AUC@30',
        'placeholder'
    ]

    summary_values = []
    summary_headers = []

    for ds_key in dataset_groups:
        mask = df['seq_name'].str.contains(ds_key)
        group_df = df[mask]

        raw_metrics_dict = group_df.to_dict(orient='list')
        merged_dict = merge_split_scene_metrics(raw_metrics_dict, TARGET_COLUMNS)
        group_df = pd.DataFrame(merged_dict)

        merge_scene_metric_csv_path = os.path.join(log_dir, f"dvgt_merge_scene_metrics_{ds_key}.csv")
        group_df.to_csv(merge_scene_metric_csv_path, index=False)

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

    summary_txt_path = os.path.join(log_dir, "dvgt_merge_scene_summary_metrics.txt")
    summary_df = pd.DataFrame([summary_values], columns=summary_headers)
    summary_df.to_csv(summary_txt_path, sep='\t', index=False, float_format='%.3f')
    print(f"Summary metrics saved to {summary_txt_path}")

    print("Evaluation Summary Table:")
    transposed_df = summary_df.T
    transposed_df.columns = ['Score']
    print("\n" + transposed_df.to_string(float_format=lambda x: "{:.4f}".format(x)))

    