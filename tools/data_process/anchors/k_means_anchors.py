import sys
import os
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from sklearn.cluster import KMeans
np.set_printoptions(suppress=True, precision=4, threshold=np.inf)

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from dvgt.datasets.transforms.sequence_utils import split_scenes_into_windows
from dvgt.utils.geometry import to_homogeneous, closed_form_inverse_se3
from dvgt.utils.trajectory import convert_pose_rdf_to_trajectory_rdf

USE_NORMLIZE = False

def parse_pose_column(pose_col) -> torch.Tensor:
    poses = np.stack(pose_col.values).reshape(-1, 3, 4)
    return torch.from_numpy(to_homogeneous(poses))

class KMeansProcessor:
    def __init__(self, dataset_configs: List[Dict], target_fps: int = 2, 
                 num_future_clusters: int = 20, num_past_clusters: int = 20):
        self.dataset_configs = dataset_configs
        self.target_fps = target_fps
        self.num_future_clusters = num_future_clusters
        self.num_past_clusters = num_past_clusters
        
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
        
        traj_rdf = convert_pose_rdf_to_trajectory_rdf(T_ego_0_ego_n).to(torch.float32)
        info = f"{ds_name}|{sid}#{indices[0]}"
        return traj_rdf, info

    def _extract_past_item(self, poses: torch.Tensor, ds_name: str, sid: str, indices):
        T_ego_1_world = closed_form_inverse_se3(poses[1:2])
        T_ego_1_ego_0 = T_ego_1_world @ poses[0:1]
        
        translation = T_ego_1_ego_0[0, :3, 3].float()
        info = f"{ds_name}|{sid}#{indices[0]}"
        return translation, info

    def run_kmeans_and_save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        files = {
            'stats': open(os.path.join(output_dir, 'statistics.txt'), 'w', buffering=1),
            'details': open(os.path.join(output_dir, 'cluster_details.txt'), 'w', buffering=1),
        }

        try:
            if self.future_data['trajs']:
                all_trajs = torch.cat(self.future_data['trajs'], dim=0).cpu().numpy()
                self._process_cluster_group(
                    data=all_trajs,
                    infos=np.array(self.future_data['infos']),
                    name="Future Trajectories",
                    n_clusters=self.num_future_clusters,
                    files=files,
                    output_dir=output_dir,
                    is_future=True
                )

            if self.past_data['poses']:
                all_poses = torch.cat(self.past_data['poses'], dim=0).cpu().numpy()
                self._process_cluster_group(
                    data=all_poses,
                    infos=np.array(self.past_data['infos']),
                    name="Past Poses (Translation)",
                    n_clusters=self.num_past_clusters,
                    files=files,
                    output_dir=output_dir,
                    is_future=False
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            for f in files.values(): f.close()

    def _process_cluster_group(self, data: np.ndarray, infos: np.ndarray, name: str, n_clusters: int,
                             files: Dict, output_dir: str, is_future: bool):
        # 2. Stats
        features = data[..., :2] if is_future else data
        stats_vec = features.reshape(-1, 2) if is_future else features

        files['stats'].write(f"=== {name} (K={n_clusters}) ===\n")
        files['stats'].write(f"Count: {data.shape[0]}\n")
        files['stats'].write(f"Max: {stats_vec.max(axis=0)}\n")
        files['stats'].write(f"Min: {stats_vec.min(axis=0)}\n")
        files['stats'].write(f"Mean: {stats_vec.mean(axis=0)}\n")
        files['stats'].write(f"Std: {stats_vec.std(axis=0)}\n\n")   

        # 3. K-Means
        if USE_NORMLIZE:
            # 聚类前，xyz分别维度执行单独的标准化，但时间轴上保持一致，因为时间轴上约远方差需要越大，远处的误差比近处的误差更重要
            normed_mean = stats_vec.mean(axis=0)
            normed_std = stats_vec.std(axis=0)
            normed_std = np.where(normed_std < 1e-6, 1e-6, normed_std)
            features = (features - normed_mean) / normed_std
            # 这个需要保存下来，后续对gt和anchor做标准化后再计算距离

        features = features.reshape(data.shape[0], 16) if is_future else features

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50, max_iter=1000).fit(features)
        centers = kmeans.cluster_centers_

        if is_future:
            centers = centers.reshape(n_clusters, 8, 2)

        if USE_NORMLIZE:
            centers = (centers * normed_std) + normed_mean
        labels = kmeans.labels_

        # 4. Save NPY
        fname = f'future_anchors_{n_clusters}_rdf.npz' if is_future else f'past_anchors_{n_clusters}_rdf.npz'
        if USE_NORMLIZE:
            np.savez(os.path.join(output_dir, fname), **{
                'centers': centers,
                'normed_mean': normed_mean,
                'normed_std': normed_std
            })
        else:
            np.savez(os.path.join(output_dir, fname), **{
                'centers': centers,
            })
        center_index = np.argsort(centers[:, -1, 0]) if is_future else np.argsort(centers[:, -1])
        files['stats'].write(f"\nCenters sorted: \n{centers[center_index]}\n\n")

        # 5. Save Details & Mapping
        files['details'].write(f"=== {name} Cluster Details (K={n_clusters}) ===\n")
        cluster_dict = {}
        for k in range(n_clusters):
            idxs = np.where(labels == k)[0]
            files['details'].write(f"\n--- Cluster {k} Count: {len(idxs)} ---\n")
            
            cluster_data = []
            for i, idx in enumerate(idxs):
                val = data[idx]
                info = infos[idx]
                cluster_data.append({'val': val, 'info': info})
                if i < 5: files['details'].write(f"  {info}\n")

            files['details'].write(f"\nCenters: \n{centers[k]}\n\n")

            cluster_dict[k] = cluster_data
            
        pkl_name = f'future_cluster_mapping_{n_clusters}_rdf.pkl' if is_future else f'past_cluster_mapping_{n_clusters}_rdf.pkl'
        with open(os.path.join(output_dir, pkl_name), 'wb') as f:
            pickle.dump(cluster_dict, f)

        # 6. Visualization
        if is_future:
            self._plot_future_2d(centers, output_dir, n_clusters)
        else:
            self._plot_past_3d(centers, output_dir, n_clusters)

    def _plot_future_2d(self, anchors, output_dir, k):
        plt.figure(figsize=(10, 10))
        for i in range(len(anchors)):
            traj_x = np.concatenate(([0], anchors[i, :, 0]))
            traj_y = np.concatenate(([0], anchors[i, :, 1]))
            plt.plot(traj_x, traj_y, marker='.', label=f'C{i}')
        plt.title(f'Future Trajectory Anchors (K={k})')
        plt.grid(True); plt.axis('equal')
        plt.savefig(os.path.join(output_dir, f'vis_future_anchors_{k}.png'))
        plt.close()

    def _plot_past_3d(self, anchors, output_dir, k):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, 0, 0, c='black', s=100, label='Current')
        ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], c='red', marker='^', s=50, label='Past Anchors')
        for i in range(len(anchors)):
            ax.plot([0, anchors[i, 0]], [0, anchors[i, 1]], [0, anchors[i, 2]], 'k-', alpha=0.3)
        ax.set_title(f'Past Pose Translation Anchors (K={k})')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=20., azim=-35)
        plt.savefig(os.path.join(output_dir, f'vis_past_anchors_3d_{k}.png'))
        plt.close()

"""
python tools/data_process/anchors/k_means_anchors.py
"""

configs = [
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/kitti_train.parquet', 'original_fps': 10},
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/ddad_train.parquet', 'original_fps': 10},
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/openscene_total_train.parquet', 'original_fps': 2},
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/nuscene_train.parquet', 'original_fps': 2},
    {'parquet_path': 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/waymo_train.parquet', 'original_fps': 10},
]
output_dir = 'data_annotation/anchors/train_total_20'

if __name__ == "__main__":
    matplotlib.use('Agg')
    
    processor = KMeansProcessor(
        configs, 
        target_fps=2, 
        num_future_clusters=20, 
        num_past_clusters=20    
    )
    
    for conf in configs:
        processor.process_dataset(conf)
    
    processor.run_kmeans_and_save(output_dir=output_dir)