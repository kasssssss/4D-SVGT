"""
 dpg使用注释
 1. pose代表：sensor to world
 2. extinsics代表：sensor to vehicle
"""
import numpy as np
from pathlib import Path
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging
import torch

from .factory import register_dataset
from .base_dataset import BaseDataset
from utils.image import process_image_sequentially, resize_to_aspect_ratio, get_fov_x_deg, sample_sparse_depth
from dvgt.utils.geometry import closed_form_inverse_se3
from utils.transformer_np import project_lidar_to_depth
from utils.tools import gen_and_create_output_dirs

logger = logging.getLogger(__name__)

@register_dataset('ddad')
class DdadDataset(BaseDataset):
    CAMERA_NAMES_MAP = {
        'camera_01': 0, 
        'camera_05': 1, 
        'camera_06': 2, 
        'camera_07': 3, 
        'camera_08': 4, 
        'camera_09': 5
    }

    def __init__(
        self,
        data_root: str = 'public_datasets/ddad/ddad_train_val/ddad.json',
        storage_pred_depth_path: str = 'public_datasets/data_annotation/pred_depth/moge_v2_large/ddad',
        storage_align_depth_path: str = 'public_datasets/data_annotation/align_depth/moge_v2_large_correct_focal/ddad',
        storage_proj_depth_path: str = 'public_datasets/data_annotation/proj_depth/moge_v2_large_correct_focal/ddad',
        storage_image_path: str = 'public_datasets/data_annotation/image/moge_v2_large_correct_focal/ddad',
        num_tokens: int = 2500,
        split: str = 'val',  # 'train' or 'val'
    ) -> None:
        self.dgp_dataset = SynchronizedSceneDataset(
            data_root,
            split=split,
            datum_names=list(self.CAMERA_NAMES_MAP.keys()) + ["lidar"],
            only_annotated_datums=True,
        )
        super().__init__(data_root, storage_pred_depth_path, storage_proj_depth_path, storage_image_path, storage_align_depth_path, num_tokens)
        self.data_root = self.data_root.parent

    @dataclass(frozen=True, slots=True)
    class SampleData:
        dgp_idx: int
        scene_id: str
        frame_idx: int
        cam_type: str
        filename: Path

    def _gather_samples(self) -> List[SampleData]:
        samples_to_process = []

        for dgp_idx in range(len(self.dgp_dataset)):
            scene_idx, frame_idx, _ = self.dgp_dataset.dataset_item_index[dgp_idx]

            for cam_type in self.CAMERA_NAMES_MAP.keys():
                datum_info = self.dgp_dataset.get_datum(scene_idx, frame_idx, cam_type)
                filename = Path(f"{scene_idx:06d}") / datum_info.datum.image.filename
                
                samples_to_process.append(self.SampleData(
                    dgp_idx=dgp_idx,
                    scene_id=str(scene_idx),
                    frame_idx=frame_idx,
                    cam_type=cam_type,
                    filename=filename,
                ))

        logging.info(f"处理：{len(samples_to_process)}条数据")
        return samples_to_process

    def _get_image_and_meta_info(self, sample_dgp: Dict, item: SampleData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam_info = sample_dgp[self.CAMERA_NAMES_MAP[item.cam_type]]
        
        image = np.asarray(cam_info['rgb'])
        original_aspect_ratio = image.shape[1] / image.shape[0]

        distortion = cam_info['distortion']
        distortion = np.array([distortion['k1'], distortion['k2'], distortion['p1'], distortion['p2'], distortion['k3']])
        if (distortion == 0).all():
            distortion = None

        cam_intrinsics = cam_info['intrinsics'].astype(np.float64)

        image_input, cam_intrinsics_input, _ = process_image_sequentially(image, cam_intrinsics, distortion, num_tokens=self.num_tokens)

        image_save, cam_intrinsics_save = resize_to_aspect_ratio(image_input, cam_intrinsics_input, original_aspect_ratio)

        cam_to_world = cam_info['pose'].matrix.astype(np.float64)
        cam_extrinsics = closed_form_inverse_se3(cam_to_world)
        
        # ego to world
        lidar_meta = sample_dgp[-1]
        lidar_to_world = lidar_meta['pose'].matrix.astype(np.float64)
        lidar_to_ego = lidar_meta['extrinsics'].matrix.astype(np.float64)
        ego_to_lidar = closed_form_inverse_se3(lidar_to_ego)
        ego_to_world = lidar_to_world @ ego_to_lidar
        T_world_ego_rdf = ego_to_world @ self.T_flu_rdf
        
        return (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, cam_extrinsics[:3], T_world_ego_rdf[:3]
        )

    def _get_proj_depth(
        self, 
        sample_dgp: Dict, 
        cam_intrinsics: np.ndarray, 
        cam_extrinsics: np.ndarray, 
        height: int, width: int
    ):
        lidar_meta = sample_dgp[-1]
        lidar_in_world = lidar_meta['pose'] * lidar_meta['point_cloud']     # in world coord
        
        proj_depth = project_lidar_to_depth(
            points=lidar_in_world, 
            points_coordinate_to_camera=cam_extrinsics,
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

        sample_dgp = self.dgp_dataset[item.dgp_idx][0]
        
        (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, cam_extrinsics, ego_to_world
        ) = self._get_image_and_meta_info(sample_dgp, item)

        data_dict.update({
            'scene_id': item.scene_id,
            'frame_idx': item.frame_idx,
            'cam_type': item.cam_type,
            'filename': str(item.filename.with_suffix('')),
            'intrinsics': cam_intrinsics_save.flatten().tolist(),
            'extrinsics': cam_extrinsics.flatten().tolist(),
            'ego_to_world': ego_to_world.flatten().tolist(),
        })

        H_save, W_save = image_save.shape[:2]
        H_input, W_input = image_input.shape[:2]

        fov_x_deg = get_fov_x_deg(W_input, cam_intrinsics_input[0, 0])
        
        # MoGe input
        data_dict.update({
            'image_input': torch.from_numpy(image_input).to(torch.float32),
            'fov': fov_x_deg,    # float
        })
        
        proj_depth = self._get_proj_depth(sample_dgp, cam_intrinsics_save, cam_extrinsics, H_save, W_save)

        data_dict.update({
            'proj_depth': torch.from_numpy(proj_depth).to(torch.float32),
            'image_save': image_save,     # 后续可以直接保存这个 uint8 的图片
            'image_size': (H_input, W_input)
        })

        return data_dict