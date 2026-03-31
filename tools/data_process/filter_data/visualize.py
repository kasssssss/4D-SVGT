import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import sys
import argparse

def generate_plots_for_columns(df, plot_configs, base_output_dir):
    """
    为DataFrame中指定的多个列绘制直方图、箱线图和CDF图。
    每个列的图表将保存在以该列名命名的单独子目录中。

    Args:
        df (pd.DataFrame): 包含数据的DataFrame。
        plot_configs (dict): 一个字典，键是列名，值是需要在CDF图上标注的百分位列表。
        base_output_dir (Path): 保存所有子目录和绘图的根目录路径对象。
    """
    # 设置Seaborn的绘图风格
    sns.set_theme(style="whitegrid")

    for metric_name, percentiles in plot_configs.items():
        # 检查列是否存在于DataFrame中
        if metric_name not in df.columns:
            print(f"警告: 列 '{metric_name}' 不在DataFrame中，已跳过。", file=sys.stderr)
            continue

        print(f"--- 正在为 '{metric_name}' 生成图表 ---")

        # 提取数据并移除任何可能存在的NaN值
        metric_data = df[metric_name].dropna()
        if metric_data.empty:
            print(f"警告: 列 '{metric_name}' 中没有有效数据，已跳过。", file=sys.stderr)
            continue

        # --- 1. 创建该列专属的输出子目录和安全的文件名 ---
        # 使用正则表达式创建一个对操作系统安全的文件/目录名
        safe_name = re.sub(r'[^\w\s-]', '', metric_name).strip().replace(' ', '_').replace('<', 'lt')
        column_output_dir = base_output_dir / safe_name
        column_output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 2. 绘制直方图 (Histogram) ---
        plt.figure(figsize=(12, 7))
        sns.histplot(metric_data, bins=10000, kde=True) # 使用较少的bins以获得更平滑的视图
        plt.title(f'Distribution of {metric_name}', fontsize=18)
        plt.xlabel(metric_name, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        hist_path = column_output_dir / f'{safe_name}_histogram.png'
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - 已保存直方图到: {hist_path}")

        # --- 3. 绘制箱线图 (Box Plot) ---
        plt.figure(figsize=(12, 7))
        sns.boxplot(x=metric_data)
        plt.title(f'Box Plot of {metric_name}', fontsize=18)
        plt.xlabel(metric_name, fontsize=14)
        box_path = column_output_dir / f'{safe_name}_boxplot.png'
        plt.savefig(box_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - 已保存箱线图到: {box_path}")

        # --- 4. 绘制累积分布函数 (CDF) ---
        plt.figure(figsize=(12, 7))
        sorted_data = np.sort(metric_data)
        cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        plt.plot(sorted_data, cumulative_prob, linewidth=2, label='CDF')
        plt.title(f'CDF of {metric_name}', fontsize=18)
        plt.xlabel(metric_name, fontsize=14)
        plt.ylabel('Cumulative Probability', fontsize=14)
        plt.grid(True, which="both", ls="--")
        plt.ylim(0, 1.05)
        
        # 添加指定的百分位辅助线和标签
        for p in percentiles:
            if 0 < p < 1:
                value = metric_data.quantile(p)
                plt.axhline(y=p, color='r', linestyle='--', linewidth=1)
                plt.axvline(x=value, color='g', linestyle='--', linewidth=1)
                plt.text(value, p + 0.01, f' {p*100:.0f}%: {value:.4f}', color='black', 
                         ha='left', va='bottom', fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.legend()
        cdf_path = column_output_dir / f'{safe_name}_cdf.png'
        plt.savefig(cdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - 已保存 CDF 图到: {cdf_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--split", default="val")
    parser.add_argument("--csv_root_path", default="output/data_process/filter_data/moge_pred_depth")
    args = parser.parse_args()

    csv_path = Path(args.csv_root_path) / args.dataset_name / args.split / "eval_results_all.csv"
    output_dir = csv_path.parent / 'visualization'

    df = pd.read_csv(csv_path, low_memory=False) 
    
    # ==================== 可配置区域 ====================
    # scale,shift,
    # Abs Rel,δ < 1.25,valid_ratio,
    # pred_depth_valid_num,proj_depth_valid_num,
    # pred_depth_max,pred_depth_min,frame_idx,cam_type,filename,image_h,image_w,
    # pred_depth_mean,proj_depth_max,proj_depth_min,proj_depth_mean
    # align_depth_max,align_depth_min,align_depth_mean,scene_id,

    plot_configurations = {
        # 'proj_depth_uv_std': [0.001, 0.01, 0.02, 0.05, 0.1],
        # 'scale': [0.01, 0.02, 0.05, 0.1, 0.9, 0.95, 0.98, 0.99],
        # 'shift': [0.01, 0.02, 0.05, 0.1, 0.9, 0.95, 0.98, 0.99],
        'Abs Rel': [0.85, 0.90, 0.95, 0.98, 0.99],
        'δ < 1.25': [0.01, 0.02, 0.05, 0.1, 0.15],
        # 'valid_ratio': [0.01, 0.02, 0.05, 0.1],
        # 'pred_depth_valid_num': [0.01, 0.02, 0.05, 0.1],
        # 'proj_depth_valid_num': [0.01, 0.02, 0.05, 0.1],
    }
    # ====================================================
    
    # 调用绘图函数
    generate_plots_for_columns(df, plot_configurations, Path(output_dir))
    
    print("--- 所有任务完成 ---")

if __name__ == '__main__':
    main()