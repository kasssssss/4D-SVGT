# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import pickle
import os
import numpy as np
import pandas as pd
from nuscenes.eval.common.utils import Quaternion
import json
from os import path as osp
from planning_utils import PlanningMetric
import torch
from tqdm import tqdm
import threading
import re
from pathlib import Path

def append_tangent_directions(traj):
    directions = []
    directions.append(np.arctan2(traj[0][1], traj[0][0]))
    for i in range(1, len(traj)):
        vector = traj[i] - traj[i-1]
        angle = np.arctan2(vector[1], vector[0])
        directions.append(angle)
    directions = np.array(directions).reshape(-1, 1)
    traj_yaw = np.concatenate([traj, directions], axis=-1)
    return traj_yaw

def print_progress(current, total):
    percentage = (current / total) * 100
    print(f"\rProgress: {current}/{total} ({percentage:.2f}%)", end="")

def process_data(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric):
    ego_boxes = np.array([[0.5 + 0.985793, 0.0, 0.0, 4.08, 1.85, 0.0, 0.0, 0.0, 0.0]])
    for i in range(start, end):
        try:
            data = key_infos['infos'][i]
            if data['token'] not in preds.keys():
                continue
            pred_traj = preds[data['token']]
            gt_traj, mask = data['gt_planning'], data['gt_planning_mask'][0]

            gt_agent_boxes = np.concatenate([data['gt_boxes'], data['gt_velocity']], -1)
            gt_agent_feats = np.concatenate([data['gt_fut_traj'][:, :6].reshape(-1, 12), data['gt_fut_traj_mask'][:, :6], data['gt_fut_yaw'][:, :6], data['gt_fut_idx']], -1)
            bev_seg = planning_metric.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats, add_rec=True)

            e2g_r_mat = Quaternion(data['ego2global_rotation']).rotation_matrix
            e2g_t = data['ego2global_translation']
            drivable_seg = planning_metric.get_drivable_area(e2g_t, e2g_r_mat, data)
            pred_traj_yaw = append_tangent_directions(pred_traj[..., :2])
            pred_traj_mask = np.concatenate([pred_traj_yaw[..., :2].reshape(1, -1), np.ones_like(pred_traj_yaw[..., :1]).reshape(1, -1), pred_traj_yaw[..., 2:].reshape(1, -1)], axis=-1)
            ego_seg = planning_metric.get_ego_seg(ego_boxes, pred_traj_mask, add_rec=True)
            
            pred_traj = torch.from_numpy(pred_traj).unsqueeze(0)
            gt_traj = torch.from_numpy(gt_traj[..., :2])
            fut_valid_flag = mask.all()
            future_second = 3
            if fut_valid_flag:
                with lock:
                    metric_dict['samples'] += 1
                for i in range(future_second):
                    cur_time = (i+1)*2
                    ade = float(
                        sum(
                            np.sqrt(
                                (pred_traj[0, i, 0] - gt_traj[0, i, 0]) ** 2
                                + (pred_traj[0, i, 1] - gt_traj[0, i, 1]) ** 2
                            )
                            for i in range(cur_time)
                        )
                        / cur_time
                    )
                    metric_dict['l2_{}s'.format(i+1)] += ade

                    ade_single = float(
                        np.sqrt(
                            (pred_traj[0, cur_time-1, 0] - gt_traj[0, cur_time-1, 0]) ** 2
                            + (pred_traj[0, cur_time-1, 1] - gt_traj[0, cur_time-1, 1]) ** 2
                        )
                    )
                    metric_dict['l2_{}s_single'.format(i+1)] += ade_single
                    
                    obj_coll, obj_box_coll = planning_metric.evaluate_coll(pred_traj[:, :cur_time], gt_traj[:, :cur_time], torch.from_numpy(bev_seg[1:]).unsqueeze(0))
                    metric_dict['plan_obj_box_col_{}s'.format(i+1)] += obj_box_coll.max().item()

                    obj_coll_single, obj_box_coll_single = planning_metric.evaluate_coll(pred_traj[:, cur_time-1:cur_time], gt_traj[:, cur_time-1:cur_time], torch.from_numpy(bev_seg[cur_time:cur_time+1]).unsqueeze(0))
                    metric_dict['plan_obj_box_col_{}s_single'.format(i+1)] += obj_box_coll_single.max().item()
                    
                    rec_out = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[0:1] == 1)).sum() > 0
                    out_of_drivable = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[1:cur_time+1] == 1)).sum() > 0
                    if out_of_drivable and not rec_out:
                        metric_dict['plan_boundary_{}s'.format(i+1)] += 1
            pbar.update(1)
        except Exception as e:
            print(e)
            pbar.update(1)

def main(args):
    print(f"Loading CSV from {args.pred_path}...")
    df = pd.read_csv(args.pred_path)
    preds = {}
    for _, row in df.iterrows():
        seq_id = row['seq_name']
        pred_traj = np.array(json.loads(row['pred_traj'])).reshape(-1, 3)[:6, :2]
        preds[seq_id] = pred_traj

    anno_path = args.anno_path
    key_infos = pickle.load(open(osp.join(args.base_path, anno_path), 'rb'))
    anno_tokens = [info['token'] for info in key_infos['infos']]

    tokens_to_evaluate = list(set(preds.keys()) & set(anno_tokens))
    num_missing_metric_cache_tokens = len(set(preds.keys()) - set(anno_tokens))
    num_unused_metric_cache_tokens = len(set(anno_tokens) - set(preds.keys()))
    if num_missing_metric_cache_tokens > 0:
        print(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        print(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    print(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")

    metric_dict = {
        'plan_obj_box_col_1s': 0,
        'plan_obj_box_col_2s': 0,
        'plan_obj_box_col_3s': 0,
        'plan_boundary_1s':0, 
        'plan_boundary_2s':0, 
        'plan_boundary_3s':0, 
        'l2_1s': 0,
        'l2_2s': 0,
        'l2_3s': 0,
        'plan_obj_box_col_1s_single': 0,
        'plan_obj_box_col_2s_single': 0,
        'plan_obj_box_col_3s_single': 0,
        'l2_1s_single': 0,
        'l2_2s_single': 0,
        'l2_3s_single': 0,
        'samples':0,
    }

    num_threads = args.num_threads  
    total_data = len(key_infos['infos'])
    data_per_thread = total_data // num_threads
    threads = []
    lock = threading.Lock()
    pbar = tqdm(total=total_data)
    for i in range(num_threads):
        start = i * data_per_thread
        end = start + data_per_thread
        if i == num_threads - 1:
            end = total_data  
        thread = threading.Thread(target=process_data, args=(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    pbar.close()    

    output_path = Path(args.pred_path).parent / "nuscene_planning_metrics.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for k, v in metric_dict.items():
            if k != "samples":
                output = f"{k}: {v / metric_dict['samples']}"
                print(output)
                f.write(output + "\n")

if __name__ == "__main__":
    # use occworld to eval
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument('--base_path', type=str, default='public_datasets/nuscene', help='Base path to the data.')
    parser.add_argument('--pred_path', type=str, default='metrics_results.csv')
    parser.add_argument('--anno_path', type=str, default='omini_drive_nuscenes2d_ego_temporal_infos_val.pkl', help='Path to the annotation file.')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use.')

    args = parser.parse_args()
    
    planning_metric = PlanningMetric(args.base_path)
    main(args)