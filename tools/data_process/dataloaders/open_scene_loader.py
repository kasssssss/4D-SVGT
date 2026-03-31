import numpy as np
import cv2
import pickle
import open3d as o3d
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging
import torch
from tqdm import tqdm

from .factory import register_dataset
from .base_dataset import BaseDataset
from utils.transformer_np import project_lidar_to_depth, build_transform_matrix
from utils.image import process_image_sequentially, resize_to_aspect_ratio, get_fov_x_deg
from dvgt.utils.geometry import closed_form_inverse_se3
from utils.tools import gen_and_create_output_dirs

logger = logging.getLogger(__name__)

@register_dataset('openscene')
class OpenSceneDataset(BaseDataset):
    CAMERA_NAMES = [
        'CAM_F0', 'CAM_B0', 'CAM_L0', 'CAM_L1', 
        'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2'
    ]

    EXCLUDED_SCENES = {
        # pose 错的离谱
        'log-0333-scene-0001', 'log-0558-scene-0002', 'log-0761-scene-0002',
        'log-0767-scene-0002', 'log-0781-scene-0001', 'log-0801-scene-0002',

        # 原地起跳的几个场景--训练集
        'log-0973-scene-0003', 'log-1035-scene-0003', 'log-1035-scene-0004',
        'log-1214-scene-0002',

        # 原地起跳的几个场景--测试集
        'log-0046-scene-0003', 'log-0132-scene-0013'
    }

    def __init__(
        self,
        data_root: str = 'public_datasets/openscene',
        storage_pred_depth_path: str = 'public_datasets/data_annotation/pred_depth/moge_v2_large/openscene',
        storage_align_depth_path: str = 'public_datasets/data_annotation/align_depth/moge_v2_large_correct_focal/openscene',
        storage_proj_depth_path: str = 'public_datasets/data_annotation/proj_depth/moge_v2_large_correct_focal/openscene',
        storage_image_path: str = 'public_datasets/data_annotation/image/moge_v2_large_correct_focal/openscene',
        num_tokens: int = 2500,
        split: str = 'train',  # 'train' or 'val' or 'trainval_slice_i', i=1~N
        split_quantity: int = 5,    # 训练集太大了，拆分为N份，多机器快速标注
    ) -> None:
        self.split = split
        self.split_quantity = split_quantity
        self.current_split_quantity = int(split.split('_')[-1]) if split != 'test' else -1
        self.meta_infos: Dict[Tuple[str, int], Dict] = {}
        super().__init__(data_root, storage_pred_depth_path, storage_proj_depth_path, storage_image_path, storage_align_depth_path, num_tokens)
    
    @dataclass(frozen=True, slots=True)
    class SampleData:
        scene_id: str
        frame_idx: int
        frame_token: str
        cam_type: str
        filename: Path

    def _gather_samples(self) -> List[SampleData]:
        if self.split == "test":
            pkl_root_path = self.data_root / "navsim_logs" / self.split
            pkl_paths = list(pkl_root_path.glob("*.pkl"))
        else:
            pkl_root_path = self.data_root / "navsim_logs" / "trainval"
            pkl_paths = list(pkl_root_path.glob("*.pkl"))
            if self.split_quantity > 1:
                size = len(pkl_paths) // self.split_quantity
                x = self.current_split_quantity
                pkl_paths = pkl_paths[(x-1)*size : (x*size if x < self.split_quantity else len(pkl_paths))]

        samples_to_process = []
        for pkl_path in tqdm(pkl_paths, desc="加载meta数据"):
            with open(pkl_path, 'rb') as f:
                metas = pickle.load(f)
            for meta in metas:
                scene_id = meta['scene_name']
                frame_idx = meta['frame_idx']
                if scene_id in self.EXCLUDED_SCENES:
                    continue
                
                self.meta_infos[(scene_id, frame_idx)] = meta
                for cam_type in self.CAMERA_NAMES:
                    cam_data = meta['cams'][cam_type]
                    filename = Path(cam_data['data_path']).with_suffix('')
                    samples_to_process.append(self.SampleData(
                        scene_id=scene_id,
                        frame_idx=frame_idx,
                        cam_type=cam_type,
                        filename=filename,
                        frame_token=meta['token'],
                    ))

        logging.info(f"Total samples to process: {len(samples_to_process)}")
        return samples_to_process   

    def _get_image_and_meta_info(self, meta: Dict, item: SampleData) -> Tuple:
        cam_meta = meta['cams'][item.cam_type]
        image_path = self.data_root / "sensor_blobs" / "all_sensor_blobs" / cam_meta['data_path']
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        original_aspect_ratio = image.shape[1] / image.shape[0]

        image_input, cam_intrinsics_input, _ = process_image_sequentially(image, cam_meta['cam_intrinsic'], dist_coeffs=cam_meta['distortion'], num_tokens=self.num_tokens)
        
        image_save, cam_intrinsics_save = resize_to_aspect_ratio(image_input, cam_intrinsics_input, original_aspect_ratio)

        # 外参
        T_lidar_cam_rdf = np.eye(4)
        T_lidar_cam_rdf[:3, :3] = cam_meta['sensor2lidar_rotation']
        T_lidar_cam_rdf[:3, 3] = cam_meta['sensor2lidar_translation']
        T_world_cam_rdf = meta['lidar2global'] @ T_lidar_cam_rdf
        T_cam_rdf_world = closed_form_inverse_se3(T_world_cam_rdf)

        # ego pose
        T_world_ego_flu = build_transform_matrix(meta['ego2global_translation'], meta['ego2global_rotation'])
        T_world_ego_rdf = T_world_ego_flu @ self.T_flu_rdf

        # ego velocity, acc, command
        ego_dynamic_state = meta["ego_dynamic_state"]
        ego_velocity_rdf = np.array((-ego_dynamic_state[1], ego_dynamic_state[0]), dtype=np.float32)
        ego_acceleration_rdf = np.array((-ego_dynamic_state[3], ego_dynamic_state[2]), dtype=np.float32)
        driving_command = meta['driving_command']   # left, forward, right, unknown

        return (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, T_cam_rdf_world[:3], T_world_ego_rdf[:3],
            T_lidar_cam_rdf, 
            ego_velocity_rdf, ego_acceleration_rdf, driving_command
        )

    def _get_proj_depth(self, meta: Dict, cam_intrinsics: np.ndarray, T_lidar_cam: np.ndarray, height: int, width: int) -> np.ndarray:
        lidar_path = self.data_root / "sensor_blobs" / "all_sensor_blobs" / meta['lidar_path']
        pcd = o3d.io.read_point_cloud(str(lidar_path))
        lidar_points = np.asarray(pcd.points)  # (N, 3)

        T_cam_lidar = closed_form_inverse_se3(T_lidar_cam)

        proj_depth = project_lidar_to_depth(
            points=lidar_points,
            points_coordinate_to_camera=T_cam_lidar,
            camera_intrinsic=cam_intrinsics,
            height=height,
            width=width
        )

        return proj_depth

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        data_dict = {}

        # 生成保存路径
        data_dict.update(
            gen_and_create_output_dirs(
                item.filename, self.storage_pred_depth_path, self.storage_proj_depth_path, 
                self.storage_align_depth_path, self.storage_image_path
            )
        )

        # meta info
        meta = self.meta_infos[(item.scene_id, item.frame_idx)]

        (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, cam_extrinsics, ego_to_world,
            T_lidar_cam_rdf, 
            ego_velocity_rdf, ego_acceleration_rdf, driving_command
        ) = self._get_image_and_meta_info(meta, item)

        data_dict.update({
            'scene_id': item.scene_id,
            'frame_idx': item.frame_idx,
            'cam_type': item.cam_type,
            'filename': str(item.filename),
            'intrinsics': cam_intrinsics_save.flatten().tolist(),
            'extrinsics': cam_extrinsics.flatten().tolist(),
            'ego_to_world': ego_to_world.flatten().tolist(),
            'ego_velocity': ego_velocity_rdf.tolist(),
            'ego_acceleration': ego_acceleration_rdf.tolist(),
            'driving_command': driving_command.tolist(),
            'frame_token': item.frame_token,
        })
    
        H_save, W_save = image_save.shape[:2]
        H_input, W_input = image_input.shape[:2]
        
        fov_x_deg = get_fov_x_deg(W_input, cam_intrinsics_input[0, 0])
        
        # MoGe input
        data_dict.update({
            'image_input': torch.from_numpy(image_input).to(torch.float32),
            'fov': fov_x_deg,    # float
        })
        
        proj_depth = self._get_proj_depth(meta, cam_intrinsics_save, T_lidar_cam_rdf, H_save, W_save)

        data_dict.update({
            'proj_depth': torch.from_numpy(proj_depth).to(torch.float32),
            'image_save': image_save,     # 后续可以直接保存这个 uint8 的图片
            'image_size': (H_input, W_input)
        })
   
        return data_dict