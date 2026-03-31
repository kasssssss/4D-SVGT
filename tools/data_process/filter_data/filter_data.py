import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import logging
import pandas as pd
import argparse
import os
from pathlib import Path
from tools.data_process.utils.log import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--split", default="train")
    parser.add_argument("--meta_storage_path", default="data_annotation/meta/moge_v2_large_correct_focal")
    parser.add_argument("--output_meta_path", default="data_annotation/meta/moge_v2_large_correct_focal/debug1")
    parser.add_argument("--csv_root_path", default="output/data_process/filter_data/moge_pred_depth")
    return parser.parse_args()

def print_formate(row, metric_name, image_path, pred_depth_path, proj_depth_path):
    """将路径作为参数传入，避免使用全局变量"""
    logging.info(
        f"({metric_name}: {row[metric_name]:.4f})\n" + \
        f"{image_path / row['filename']}.jpg\n" + \
        f"{pred_depth_path / row['filename']}.png\n" + \
        f"{proj_depth_path / row['filename']}.png"
    )

def print_mean_metric(moge_lidar):
    # moge_lidar
    logging.info(f"moge_lidar: Abs Rel: {moge_lidar['Abs Rel'].mean():.4f}")
    logging.info(f"moge_lidar: δ < 1.25: {moge_lidar['δ < 1.25'].mean():.4f}")
    # 额外展示最大值和最小值，避免出现离谱场景
    logging.info(f"moge_lidar: Abs Rel max: {moge_lidar['Abs Rel'].max():.4f}")
    logging.info(f"moge_lidar: δ < 1.25 min: {moge_lidar['δ < 1.25'].min():.4f}")
    logging.info('#' * 20)

def print_filter_data_info(cond_name, cond, garbage_mask):
    """打印过滤数据的信息"""
    logging.info(f"--- {cond_name} ---")
    logging.info(f"该指标删除的数据量: {cond.sum() / len(cond) * 100:.2f}%")
    logging.info(f"总删除的数据量: {garbage_mask.sum() / len(garbage_mask) * 100:.2f}%")
    logging.info(f"占删除数据的比重: {cond.sum() / garbage_mask.sum() * 100:.2f}%")

def print_rebuttal_keepint_rate(cond_dict, series_index):
    stage_1_valid_point_overlap = ['cond_low_valid_ratio']
    stage_2_standard_depth_metrics = ['cond_high_abs_rel_delta']
    stage_3_alignment_quality_metrics = ['cond_high_scale', 'cond_low_scale', 'cond_high_shift', 'cond_low_shift', 'cond_low_proj_depth_valid_num', 'cond_high_proj_depth_uv_std']

    logging.info(f"--- Rebuttal data keep rate ---")
    garbage_mask = pd.Series(False, index=series_index)
    for cond_name, cond in cond_dict.items():
        if cond_name in stage_1_valid_point_overlap:
            garbage_mask |= cond
    logging.info(f"stage 1 删除的数据: {garbage_mask.sum() / len(garbage_mask) * 100:.2f}")

    for cond_name, cond in cond_dict.items():
        if cond_name in stage_2_standard_depth_metrics:
            garbage_mask |= cond
    logging.info(f"stage 2 删除的数据: {garbage_mask.sum() / len(garbage_mask) * 100:.2f}")

    for cond_name, cond in cond_dict.items():
        if cond_name in stage_3_alignment_quality_metrics:
            garbage_mask |= cond
    logging.info(f"stage 3 删除的数据: {garbage_mask.sum() / len(garbage_mask) * 100:.2f}")

threshold_dict = {
    "nuscene": {
        "Abs_Rel_upper": 0.2915,
        "δ < 1.25_down": 0.5771,
        "scale_upper": 1.3,
        "scale_down": 0.7,
        "shift_upper": 3.28,
        "shift_down": -3.56,
        "valid_ratio_down": 97.0,          # 2%
        "proj_depth_valid_num_down": 645,   # 5%
        "proj_depth_uv_std_down": 75.775,
    },
    "kitti": {
        "Abs_Rel_upper": 0.1740,
        "δ < 1.25_down": 0.7465,
        "scale_upper": 1.6221,
        "scale_down": 0.3949,
        "shift_upper": 4.0332,
        "shift_down": -6.0920,
        "valid_ratio_down": 99.54,
        "proj_depth_valid_num_down": 3371,
        "proj_depth_uv_std_down": 60,  # 1%
    },
    "ddad": {
        "Abs_Rel_upper": 0.2701,
        "δ < 1.25_down": 0.5907,
        "scale_upper": 1.6221,
        "scale_down": 0.3949,
        "shift_upper": 4.0332,
        "shift_down": -7.76,
        "valid_ratio_down": 98.52,
        "proj_depth_valid_num_down": 4136,
        "proj_depth_uv_std_down": 71,
    },
    "waymo": {
        "Abs_Rel_upper": 0.2632,
        "δ < 1.25_down": 0.4235,
        "scale_upper": 1.94,
        "scale_down": 0.5,
        "shift_upper": 12,
        "shift_down": -12,
        "valid_ratio_down": 97.39,
        "proj_depth_valid_num_down": 2731,
        "proj_depth_uv_std_down": 85.10,
    },
    "openscene": {
        "Abs_Rel_upper": 0.2571,
        "δ < 1.25_down": 0.6007,
        "scale_upper": 1.28,
        "scale_down": 0.73,
        "shift_upper": 2.37,
        "shift_down": -2.22,
        "valid_ratio_down": 98.95,
        "proj_depth_valid_num_down": 2645,
        "proj_depth_uv_std_down": 84.15,
    },
}

def main():
    args = parse_args()
    # --- 1. 初始化 ---
    dataset_threshold = threshold_dict[args.dataset_name]
    META_PARQUET_PATH = Path(args.meta_storage_path) / f"{args.dataset_name}_{args.split}.parquet"
    MOGE_LIDAR_CSV_PATH = Path(args.csv_root_path) / args.dataset_name / args.split / f"eval_results_all.csv"
    
    # 定义更新后 meta 文件的输出路径
    OUTPUT_META_PATH = Path(args.output_meta_path) / f"{args.dataset_name}_{args.split}.parquet"
    OUTPUT_META_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path(MOGE_LIDAR_CSV_PATH).parent

    setup_logger(log_name="filter_logs", log_dir=str(log_dir))

    index_cols = ['scene_id', 'frame_idx', 'cam_type']
    moge_lidar_df = pd.read_csv(MOGE_LIDAR_CSV_PATH)
    moge_lidar_df['cam_type'] = moge_lidar_df['cam_type'].astype(str)
    moge_lidar_df['scene_id'] = moge_lidar_df['scene_id'].astype(str)
    moge_lidar_df = moge_lidar_df.set_index(index_cols).sort_index()

    meta_df = pd.read_parquet(META_PARQUET_PATH)
    meta_df['cam_type'] = meta_df['cam_type'].astype(str)
    meta_df = meta_df.set_index(index_cols).sort_index()

    # 确保索引一致
    base_index = moge_lidar_df.index
    meta_matches = base_index.equals(meta_df.index)
    logging.info(f"meta_df index matches base: {meta_matches} (Length: {len(meta_df.index)})")

    # 深度调试：如果不匹配 (meta_matches is False)，看看它们到底差了什么
    if not meta_matches:
        logging.warning("--- 索引不匹配！正在查找差异... ---")
        
        # 找出在 meta_df 中，但不在 moge_lidar_df (garbage_mask) 中的索引
        # 这些索引在赋值时，因为在 garbage_mask 中找不到，所以被赋了 NaN
        in_meta_not_in_moge = meta_df.index.difference(base_index)
        logging.warning(f"Indexes in meta_df but NOT in moge_lidar_df: {len(in_meta_not_in_moge)}")
        if len(in_meta_not_in_moge) > 0:
            logging.warning(f"Example extras in meta_df: {in_meta_not_in_moge[:5]}") # 打印前5个差异

        # 找出在 moge_lidar_df (garbage_mask) 中，但不在 meta_df 中的索引
        # 这些索引在赋值时被丢弃了
        in_moge_not_in_meta = base_index.difference(meta_df.index)
        logging.warning(f"Indexes in moge_lidar_df but NOT in meta_df: {len(in_moge_not_in_meta)}")
        if len(in_moge_not_in_meta) > 0:
            logging.warning(f"Example extras in moge_lidar_df: {in_moge_not_in_meta[:5]}")

    logging.info("\n---选择出垃圾数据，需要过滤掉的数据 ---")
    cond_dict = {}
    # 1. 根据abs rel和δ < 1.25筛选
    cond_dict['cond_high_abs_rel_delta'] = (dataset_threshold['Abs_Rel_upper'] < moge_lidar_df['Abs Rel']) | (dataset_threshold['δ < 1.25_down'] > moge_lidar_df['δ < 1.25'])

    # 2. 根据scale筛选
    cond_dict['cond_high_scale'] = moge_lidar_df['scale'] > dataset_threshold['scale_upper']
    cond_dict['cond_low_scale'] = moge_lidar_df['scale'] < dataset_threshold['scale_down']
    logging.info(f"scale > {dataset_threshold['scale_upper']}, scale < {dataset_threshold['scale_down']}")

    # 3. 根据shift筛选
    cond_dict['cond_high_shift'] = moge_lidar_df['shift'] > dataset_threshold['shift_upper']
    cond_dict['cond_low_shift'] = moge_lidar_df['shift'] < dataset_threshold['shift_down']
    logging.info(f"shift > {dataset_threshold['shift_upper']}, shift < {dataset_threshold['shift_down']}")

    # 4. 根据valid_ratio筛选
    cond_dict['cond_low_valid_ratio'] = moge_lidar_df['valid_ratio'] < dataset_threshold['valid_ratio_down']
    logging.info(f"valid_ratio < {dataset_threshold['valid_ratio_down']}")

    # 5. 根据proj depth valid num筛选
    cond_dict['cond_low_proj_depth_valid_num'] = moge_lidar_df['proj_depth_valid_num'] < dataset_threshold['proj_depth_valid_num_down']
    logging.info(f"proj_depth_valid_num < {dataset_threshold['proj_depth_valid_num_down']}")

    # 6. 根据proj_depth_uv_std 筛选
    cond_dict['cond_high_proj_depth_uv_std'] = moge_lidar_df['proj_depth_uv_std'] < dataset_threshold['proj_depth_uv_std_down']
    logging.info(f"proj_depth_uv_std < {dataset_threshold['proj_depth_uv_std_down']}")


    logging.info("开始数据筛选...")
    # --- 合并条件，筛选出所有“垃圾数据”，需要过滤的数据 ---
    garbage_mask = pd.Series(False, index=moge_lidar_df.index)
    for cond_name, cond in cond_dict.items():
        garbage_mask |= cond
    for cond_name, cond in cond_dict.items():
        print_filter_data_info(cond_name, cond, garbage_mask)

    # --- 更新后的数据，计算三种指标 -- 
    print_mean_metric(moge_lidar_df[~garbage_mask])

    # 打印论文里写的三种过滤方式各删除了多少数据
    print_rebuttal_keepint_rate(cond_dict, moge_lidar_df.index)

    # --- 更新 meta 文件 ---
    logging.info("正在根据筛选结果更新 meta 文件...")
    
    # 新增列并设置
    meta_df['use_lidar_proj_depth'] = garbage_mask
        
    # 恢复索引，得到最终的 meta DataFrame
    meta_df_final = meta_df.reset_index()
    meta_df_final.to_parquet(OUTPUT_META_PATH, engine='pyarrow', compression='zstd', index=False)
    logging.info(f"已将更新后的 meta 文件保存至: {OUTPUT_META_PATH}")
    logging.info(f"删除的数量量为：{meta_df_final['use_lidar_proj_depth'].sum()}")
    
if __name__ == '__main__':
    main()