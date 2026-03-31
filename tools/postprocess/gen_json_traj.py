import pandas as pd
import json
import numpy as np
import os
from pathlib import Path

"""
python tools/postprocess/gen_json_traj.py \
    --pth output/stream_dvgt/eval/main_table/base_v3_1_openscene_navsim_e20/metrics_results.csv
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pth", type=str, default="output/stream_dvgt/eval/main_table/base_v3_openscene_navsim_e20/metrics_results.csv")
args = parser.parse_args()

pth = Path(args.pth)
df = pd.read_csv(pth)

if 'pred_traj' in df.columns and 'gt_traj' in df.columns:
    traj_dict = {}
    
    # 2. 提取这两列和 seq_name 列（通过切片，不修改原 df）
    for _, row in df[['seq_name', 'pred_traj', 'gt_traj']].iterrows():
        seq_name = str(row['seq_name'])
        
        # 将展平的轨迹数据 reshape 为 (8, 3) 
        # 使用 .tolist() 是因为 numpy array 无法直接被 json.dump 序列化
        p_traj = np.array(json.loads(row['pred_traj'])).reshape(8, 3).tolist()
        g_traj = np.array(json.loads(row['gt_traj'])).reshape(8, 3).tolist()
        
        # 4. json的格式是：{'seq_name': [pred_traj, gt_traj ]}
        traj_dict[seq_name] = [p_traj, g_traj]


    # 3. 存为一个json
    traj_json_path = pth.parent / 'trajectory.json'
    with open(traj_json_path, 'w', encoding='utf-8') as f:
        # 建议不加缩进(indent)以减小文件体积，如果需要人类可读可以加上 indent=4
        json.dump(traj_dict, f) 
    
    print(f'保存到：{traj_json_path}')
    
