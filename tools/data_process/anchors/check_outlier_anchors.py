import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
np.set_printoptions(suppress=True, precision=4, threshold=np.inf)

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from dvgt.datasets.transforms.sequence_utils import split_scenes_into_windows
from dvgt.utils.geometry import to_homogeneous, closed_form_inverse_se3
from dvgt.utils.trajectory import convert_pose_rdf_to_trajectory_flu

def parse_pose_column(pose_col) -> torch.Tensor:
    poses = np.stack(pose_col.values).reshape(-1, 3, 4)
    return torch.from_numpy(to_homogeneous(poses))

class PoseAnalyzer:
    def __init__(self, dataset_configs: List[Dict], target_fps: int = 2):
        self.dataset_configs = dataset_configs
        self.target_fps = target_fps
        
        self.future_data = {'trajs': [], 'infos': []}
        self.past_data = {'poses': [], 'infos': []}

    def process_dataset(self, config: Dict):
        path = config['parquet_path']
        dataset_name = os.path.basename(path).split('_')[0]
        original_fps = config.get('original_fps', 10)
        fps_step = max(1, original_fps // self.target_fps)

        try:
            df = pd.read_parquet(path, columns=['scene_id', 'frame_idx', 'cam_type', 'ego_to_world'])
        except Exception:
            return

        df_unique = df.groupby(['scene_id', 'frame_idx']).first().reset_index()
        scenes = {
            sid: grp.set_index(['frame_idx']).sort_index() 
            for sid, grp in df_unique.groupby('scene_id')
        }

        self._process_windows(scenes, 9, fps_step, dataset_name, is_future=True)
        self._process_windows(scenes, 2, fps_step, dataset_name, is_future=False)

    def _process_windows(self, scenes: Dict, window_size: int, step: int, ds_name: str, is_future: bool):
        windows = split_scenes_into_windows(scenes, window_size, step, ds_name)
        if not windows:
            return
            
        traj_buffer, info_buffer = [], []
        
        for key, df_win in tqdm(windows.items(), desc=f"Extr. {ds_name} {'Fut' if is_future else 'Pst'}", leave=False):
            scene_id = key.split('#')[1]
            poses = parse_pose_column(df_win['ego_to_world'])
            
            if is_future:
                traj, info = self._extract_future_item(poses, ds_name, scene_id, df_win.index)
            else:
                traj, info = self._extract_past_item(poses, ds_name, scene_id, df_win.index)
                
            traj_buffer.append(traj)
            info_buffer.append(info)

        if traj_buffer:
            batch = torch.stack(traj_buffer)
            target_data = self.future_data if is_future else self.past_data
            target_data['trajs' if is_future else 'poses'].append(batch)
            target_data['infos'].extend(info_buffer)


    def _extract_future_item(self, poses: torch.Tensor, ds_name: str, sid: str, indices):
        T_ego_0_world = closed_form_inverse_se3(poses[0:1])
        T_ego_0_ego_n = T_ego_0_world @ poses[1:]
        
        traj_flu = convert_pose_rdf_to_trajectory_flu(T_ego_0_ego_n.unsqueeze(0)).squeeze(0).float()
        info = f"{ds_name}|{sid}#{indices[0]}"
        return traj_flu, info

    def _extract_past_item(self, poses: torch.Tensor, ds_name: str, sid: str, indices):
        T_ego_1_world = closed_form_inverse_se3(poses[1:2])
        T_ego_1_ego_0 = T_ego_1_world @ poses[0:1]
        
        translation = T_ego_1_ego_0[0, :3, 3].float()
        info = f"{ds_name}|{sid}#{indices[0]}"
        return translation, info

    def run_analysis(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        traj_outlier_file = os.path.join(output_dir, 'analysis_outliers_traj.txt')
        translation_outlier_file = os.path.join(output_dir, 'analysis_outliers_translation.txt')
        
        print(f"\nStarting analysis... Results will be saved to {output_dir}")

        with open(traj_outlier_file, 'w') as f_out:
            
            # ==========================================
            # 1. Analyze Future Trajectories (FLU)
            # ==========================================
            if self.future_data['trajs']:
                print("Analyzing Future Trajectories...")
                # Concat all: (Total_Samples, 8, 2)
                all_trajs = torch.cat(self.future_data['trajs'], dim=0).cpu().numpy()
                all_infos = np.array(self.future_data['infos'])
                
                # Plot Histograms
                # Flatten to see distribution of all points
                self._plot_histogram(all_trajs[:, :, 0].flatten(), 'Future Traj X (Longitudinal/Front)', 
                                   'Distance (m)', plot_dir, 'hist_future_x.png')
                self._plot_histogram(all_trajs[:, :, 1].flatten(), 'Future Traj Y (Lateral/Left)', 
                                   'Distance (m)', plot_dir, 'hist_future_y.png')
                
                # Check Outliers
                f_out.write("##### Future Trajectory Outliers (FLU) #####\n")
                f_out.write("Thresholds: X > 140 or X < -2 | Y > 45 or Y < -45\n\n")
                
                count = 0
                for i in range(len(all_trajs)):
                    traj = all_trajs[i]
                    xs, ys = traj[:, 0], traj[:, 1]
                    
                    # 阈值判定
                    if (xs.max() > 140 or xs.min() < -2 or 
                        ys.max() > 45 or ys.min() < -45):
                        
                        f_out.write(f"ID: {all_infos[i]}\n")
                        f_out.write(f"Val:\n{traj}\n\n")
                        count += 1
                
                summary = f"Future Outliers: {count} / {len(all_trajs)} ({count/len(all_trajs)*100:.2f}%)"
                f_out.write(f"{summary}\n\n")
                print(summary)

        with open(translation_outlier_file, 'w') as f_out:
            # ==========================================
            # 2. Analyze Past Translations (RDF)
            # ==========================================
            if self.past_data['poses']:
                print("Analyzing Past Translations...")
                # Concat all: (Total_Samples, 3)
                all_trans = torch.cat(self.past_data['poses'], dim=0).cpu().numpy()
                all_infos = np.array(self.past_data['infos'])
                
                # Plot Histograms
                self._plot_histogram(all_trans[:, 0], 'Past Trans X (Lateral/Right)', 
                                   'Delta (m)', plot_dir, 'hist_past_x.png')
                self._plot_histogram(all_trans[:, 1], 'Past Trans Y (Vertical/Down)', 
                                   'Delta (m)', plot_dir, 'hist_past_y.png')
                self._plot_histogram(all_trans[:, 2], 'Past Trans Z (Longitudinal/Front)', 
                                   'Delta (m)', plot_dir, 'hist_past_z.png')
                
                # Check Outliers
                f_out.write("##### Past Translation Outliers (RDF) #####\n")
                f_out.write("Thresholds: X > 2.0 or X < -2.0 | Y > 1.5 or Y < -1.5 | Z > 20 or Z < -1.0\n\n")
                
                count = 0
                for i in range(len(all_trans)):
                    val = all_trans[i]
                    x, y, z = val[0], val[1], val[2]
                    
                    # 阈值判定 (基于物理推算的修正值)
                    if (x > 2.0 or x < -2.0 or 
                        y > 1.5 or y < -1.5 or 
                        z > 1.0 or z < -19.0):
                        
                        f_out.write(f"ID: {all_infos[i]}\n")
                        f_out.write(f"Val: {val}\n\n")
                        count += 1
                
                summary = f"Past Outliers: {count} / {len(all_trans)} ({count/len(all_trans)*100:.2f}%)"
                f_out.write(f"{summary}\n\n")
                print(summary)

    def _plot_histogram(self, data, title, xlabel, output_dir, filename):
        """
        绘制对数坐标直方图
        """
        plt.figure(figsize=(10, 6))
        # log=True 关键参数，用于显示稀疏的异常值
        plt.hist(data, bins=100, log=True, color='skyblue', edgecolor='black', alpha=0.7)
        
        # 统计信息展示在标题中
        stats_str = f"Min: {data.min():.3f}, Max: {data.max():.3f}, Mean: {data.mean():.3f}"
        plt.title(f"{title}\n{stats_str}")
        
        plt.xlabel(xlabel)
        plt.ylabel("Count (Log Scale)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()


"""
We leverage Parquet for efficient Pose Anomaly Detection. 
Update the 'parquet_path' below to your local directory to screen for outliers and inconsistencies.
"""

# train
configs = [
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/kitti_train.parquet', 'original_fps': 10},
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/ddad_train.parquet', 'original_fps': 10},
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/openscene_total_train.parquet', 'original_fps': 2},
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/nuscene_train.parquet', 'original_fps': 2},
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/waymo_train.parquet', 'original_fps': 10},
]

# val
# configs = [
#     {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/kitti_val.parquet', 'original_fps': 10},
#     {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/ddad_val.parquet', 'original_fps': 10},
#     {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/openscene_test.parquet', 'original_fps': 2},
#     {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/nuscene_val.parquet', 'original_fps': 2},
#     {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/waymo_val.parquet', 'original_fps': 10},
# ]

output_dir = 'data_annotation/anchors/check_outlier_anchors'

if __name__ == "__main__":
    matplotlib.use('Agg')
    
    analyzer = PoseAnalyzer(
        configs, 
        target_fps=2
    )
    
    for conf in configs:
        analyzer.process_dataset(conf)

    analyzer.run_analysis(output_dir=output_dir)