"""
cam: rdf
imu(ego pose, vel, acc): flu
lidar: rfu
"""
import os
import numpy as np
import cv2
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils import splits
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging
import torch
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import pickle

from .factory import register_dataset
from .base_dataset import BaseDataset
from tools.data_process.utils.image import process_image_sequentially, resize_to_aspect_ratio, get_fov_x_deg
from dvgt.utils.geometry import closed_form_inverse_se3
from tools.data_process.utils.transformer_np import project_lidar_to_depth, build_transform_matrix
from tools.data_process.utils.tools import gen_and_create_output_dirs

logger = logging.getLogger(__name__)

@register_dataset('nuscene')
class NusceneDataset(BaseDataset):
    CAMERA_NAMES = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
    ]

    def __init__(
        self,
        data_root: str = 'public_datasets/nuscenes',
        storage_pred_depth_path: str = 'public_datasets/data_annotation/pred_depth/moge_v2_large/nuscenes',
        storage_align_depth_path: str = 'public_datasets/data_annotation/align_depth/moge_v2_large_correct_focal/nuscenes',
        storage_proj_depth_path: str = 'public_datasets/data_annotation/proj_depth/moge_v2_large_correct_focal/nuscenes',
        storage_image_path: str = 'public_datasets/data_annotation/image/moge_v2_large_correct_focal/nuscenes',
        extra_meta_path: str = 'public_datasets/nuscenes_occ_world_pkl',
        num_tokens: int = 2500,
        version: str = "v1.0-trainval",
        split: str = 'train'    # 'train' or 'val'
    ) -> None:
        self._nusc = None
        self._nusc_can = None
        self.version = version
        self.scene_names = splits.train if split == 'train' else splits.val
        self.scene_names = set(self.scene_names)
        extra_meta_path = os.path.join(extra_meta_path, 'nuscenes_infos_train_temporal_v3_scene.pkl' if split == 'train' else 'nuscenes_infos_val_temporal_v3_scene.pkl')
        with open(extra_meta_path, 'rb') as f:
            self.extra_meta = pickle.load(f)['infos']

        super().__init__(data_root, storage_pred_depth_path, storage_proj_depth_path, storage_image_path, storage_align_depth_path, num_tokens)
    
    @property
    def nusc(self):
        """ Lazy load nusc object for multiprocessing compatibility. """
        if self._nusc is None:
            self._nusc = NuScenes(self.version, str(self.data_root), verbose=False)
        return self._nusc

    @property
    def nusc_can(self):
        """ Lazy load nusc object for multiprocessing compatibility. """
        if self._nusc_can is None:
            self._nusc_can = NuScenesCanBus(dataroot=self.data_root)
        return self._nusc_can

    # _gather_samples return info class
    @dataclass(frozen=True, slots=True)
    class SampleData:
        scene_id: str
        frame_idx: int
        cam_type: str
        sample: Dict
        filename: Path
        cam_data: Dict
        frame_token: str

    def _gather_samples(self) -> List[SampleData]:
        samples_to_process = []

        for scene in self.nusc.scene:
            scene_id = scene['name']
            if scene_id not in self.scene_names:
                continue
            sample_token = scene['first_sample_token']
            frame_idx = 0
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                for cam_name in self.CAMERA_NAMES:
                    cam_token = sample['data'][cam_name]
                    cam_data = self.nusc.get('sample_data', cam_token)
                    filename = Path(cam_data['filename']).with_suffix('')
                    samples_to_process.append(self.SampleData(
                        scene_id=scene_id,
                        frame_idx=frame_idx,
                        cam_type=cam_name,
                        sample=sample,
                        filename=filename,
                        cam_data=cam_data,
                        frame_token=sample_token,
                    ))
                sample_token = sample['next']
                frame_idx += 1
                
        logging.info(f"处理：{len(samples_to_process)}条数据")
        return samples_to_process
    
    def _get_image_and_meta_info(self, cam_data: Dict, scene_name: str, frame_idx: int) -> Tuple:
        image_path = self.data_root / cam_data['filename']
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        original_aspect_ratio = image.shape[1] / image.shape[0]

        cam_calib_data = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_intrinsics = np.array(cam_calib_data['camera_intrinsic'])
        image_input, cam_intrinsics_input, _ = process_image_sequentially(image, cam_intrinsics, num_tokens=self.num_tokens)
        
        image_save, cam_intrinsics_save = resize_to_aspect_ratio(image_input, cam_intrinsics_input, original_aspect_ratio)

        # 外参
        ego_pose_data = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        T_world_flu_ego_flu = build_transform_matrix(ego_pose_data['translation'], ego_pose_data['rotation'])
        T_ego_flu_cam_rdf = build_transform_matrix(cam_calib_data['translation'], cam_calib_data['rotation'])
        T_world_flu_cam_rdf = T_world_flu_ego_flu @ T_ego_flu_cam_rdf

        # 约定cam和ego都是rdf表示，world不处理，保持数据集自身的即可
        T_cam_rdf_world_flu = closed_form_inverse_se3(T_world_flu_cam_rdf)
        T_world_flu_ego_rdf = T_world_flu_ego_flu @ self.T_flu_rdf

        # acc, velocity, 有的场景没有速度加速度，直接扔掉
        ego_velocity_rdf = np.zeros(2, dtype=np.float32)
        ego_acceleration_rdf = np.zeros(2, dtype=np.float32)
        driving_command = np.append(self.extra_meta[scene_name][frame_idx]['pose_mode'], 0)
        
        try:
            pose_msgs = self.nusc_can.get_messages(scene_name, 'pose')
            if pose_msgs:
                # 寻找与当前相机最近的msg
                closest_msg = min(pose_msgs, key=lambda msg: abs(msg['utime'] - cam_data['timestamp']))
                
                # nuScenes 速度加速度的坐标系是flu
                vel = closest_msg['vel']     # [v_x, v_y, v_z]
                accel = closest_msg['accel'] # [a_x, a_y, a_z]                
                ego_velocity_rdf = np.array((-vel[1], vel[0]), dtype=np.float32)
                ego_acceleration_rdf = np.array((-accel[1], accel[0]), dtype=np.float32)
                
        except Exception as e:
            # 极少数场景可能缺少 CAN 数据包，做容错处理
            logger.warning(f"Failed to get CAN bus info for scene {scene_name}: {e}")

        return (
            cam_intrinsics_input, image_input,  # moge输入用
            cam_intrinsics_save, image_save, T_cam_rdf_world_flu[:3], T_world_flu_ego_rdf[:3],       # meta信息
            T_ego_flu_cam_rdf,      # lidar proj用
            ego_velocity_rdf, ego_acceleration_rdf, driving_command
        )

    def _get_proj_depth(self, item: SampleData, cam_intrinsics: np.ndarray, T_ego_flu_cam_rdf: np.ndarray, height: int, width: int) -> np.ndarray:
        lidar_token = item.sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_pc_path = self.data_root / lidar_data['filename']
        lidar_pc = LidarPointCloud.from_file(str(lidar_pc_path))

        lidar_calib_data = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        T_ego_flu_lidar = build_transform_matrix(lidar_calib_data['translation'], lidar_calib_data['rotation'])
        T_cam_rdf_ego_flu = closed_form_inverse_se3(T_ego_flu_cam_rdf)
        T_cam_rdf_lidar = T_cam_rdf_ego_flu @ T_ego_flu_lidar   # 这里lidar会被转化到cam坐标系，所以lidar是flu还是rdf不重要

        proj_depth = project_lidar_to_depth(
            points=lidar_pc.points[:3, :].T,
            points_coordinate_to_camera=T_cam_rdf_lidar,
            camera_intrinsic=cam_intrinsics,
            height=height, width=width
        )
        return proj_depth

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        data_dict = dict()

        # 生成保存路径
        data_dict.update(
            gen_and_create_output_dirs(
                item.filename, self.storage_pred_depth_path, self.storage_proj_depth_path, 
                self.storage_align_depth_path, self.storage_image_path
            )
        )

        # meta info
        (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, cam_extrinsics, ego_to_world,
            T_ego_flu_cam_rdf,
            ego_velocity_rdf, ego_acceleration_rdf, driving_command
        ) = self._get_image_and_meta_info(item.cam_data, item.scene_id, item.frame_idx)

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

        # MoGe input
        H_save, W_save = image_save.shape[:2]
        H_input, W_input = image_input.shape[:2]

        # 输入MoGe计算的Focal，这里fx=fy，计算fov_x
        fov_x_deg = get_fov_x_deg(W_input, cam_intrinsics_input[0, 0])
        
        data_dict.update({
            'image_input': torch.from_numpy(image_input).to(torch.float32),
            'fov': fov_x_deg,    # float
        })

        # lidra depth和其他
        proj_depth = self._get_proj_depth(item, cam_intrinsics_save, T_ego_flu_cam_rdf, H_save, W_save)

        data_dict.update({
            'proj_depth': torch.from_numpy(proj_depth).to(torch.float32),
            'image_save': image_save,     # 后续可以直接保存这个 uint8 的图片
            'image_size': (H_input, W_input)
        })
    
        return data_dict