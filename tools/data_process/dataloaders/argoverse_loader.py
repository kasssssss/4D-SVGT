import numpy as np
import cv2
import torch
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .factory import register_dataset
from .base_dataset import BaseDataset
from tools.data_process.utils.image import process_image_sequentially, resize_to_aspect_ratio, get_fov_x_deg
from tools.data_process.utils.tools import gen_and_create_output_dirs
from dvgt.utils.geometry import closed_form_inverse_se3

logger = logging.getLogger(__name__)

@register_dataset('argoverse')
class ArgoverseDataset(BaseDataset):
    """
    Argoverse 2 Dataset loading from pre-processed data.
    Structure expected:
        root/
          scene_id/
            jpg/camera_name/timestamp.jpg
            depth/camera_name/timestamp_depth.npy
            cam_parameters/camera_name/timestamp_cam.npz
    """
    
    # 你脚本中处理的传感器列表
    CAMERA_NAMES = [
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    ]

    def __init__(
        self,
        data_root: str = 'public_datasets/argoverse2_process',
        storage_pred_depth_path: str = 'public_datasets/data_annotation/pred_depth/moge_v2_large/argoverse',
        storage_align_depth_path: str = 'public_datasets/data_annotation/align_depth/moge_v2_large_correct_focal/argoverse',
        storage_proj_depth_path: str = 'public_datasets/data_annotation/proj_depth/moge_v2_large_correct_focal/argoverse',
        storage_image_path: str = 'public_datasets/data_annotation/image/moge_v2_large_correct_focal/argoverse',
        num_tokens: int = 2500,
        split: str = 'test'
    ) -> None:

        data_root = str(Path(data_root) / split)
        super().__init__(data_root, storage_pred_depth_path, storage_proj_depth_path, storage_image_path, storage_align_depth_path, num_tokens)

    @dataclass(frozen=True, slots=True)
    class SampleData:
        scene_id: str
        frame_idx: int
        cam_type: str
        timestamp: str
        image_path: Path
        depth_path: Path
        param_path: Path
        filename: Path # 相对路径，用于生成输出目录

    def _gather_samples(self) -> List[SampleData]:
        """
        遍历预处理后的目录结构，收集所有样本
        """
        samples_to_process = []
        
        # data_root 下面是各个 scene_id 文件夹
        scene_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]

        for scene_dir in scene_dirs:
            scene_id = scene_dir.name
            
            for cam_name in self.CAMERA_NAMES:
                jpg_dir = scene_dir / "jpg" / cam_name
                depth_dir = scene_dir / "depth" / cam_name
                param_dir = scene_dir / "meta_parameters" / cam_name
                
                if not jpg_dir.exists():
                    continue

                # 获取所有图片
                image_files = sorted(jpg_dir.glob("*.jpg"))
                
                for idx, img_path in enumerate(image_files):
                    timestamp = img_path.stem # 比如 315967841002025000
                    
                    # 构造对应的 depth 和 param 路径
                    depth_path = depth_dir / f"{timestamp}_depth.npy"
                    param_path = param_dir / f"{timestamp}_cam.npz"
                    
                    # 确保数据完整
                    if not depth_path.exists() or not param_path.exists():
                        continue

                    # 构造相对路径文件名 (用于生成 output 目录结构)
                    # 格式: scene_id/camera_name/timestamp
                    filename = Path(scene_id) / cam_name / timestamp

                    samples_to_process.append(self.SampleData(
                        scene_id=scene_id,
                        frame_idx=idx,
                        cam_type=cam_name,
                        timestamp=timestamp,
                        image_path=img_path,
                        depth_path=depth_path,
                        param_path=param_path,
                        filename=filename
                    ))

        logging.info(f"处理 Argoverse (Pre-processed) 数据：{len(samples_to_process)} 条")
        return samples_to_process

    def _get_image_and_meta_info(self, item: SampleData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image = cv2.cvtColor(cv2.imread(str(item.image_path)), cv2.COLOR_BGR2RGB)

        with np.load(item.param_path) as data:
            K_ori = data['camera_intrinsics']
            T_cam_rdf_world = data['world_to_cam']
            T_world_ego_flu = data['ego_to_world']

        depth_gt = np.load(str(item.depth_path))
        
        T_world_ego_rdf = T_world_ego_flu @ self.T_flu_rdf

        return (
            K_ori, image,
            K_ori, image,
            T_cam_rdf_world[:3], # 3x4
            T_world_ego_rdf[:3],   # 3x4
            depth_gt
        )

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

        # 获取数据
        (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, cam_extrinsics, ego_to_world,
            depth_gt
        ) = self._get_image_and_meta_info(item)

        data_dict.update({
            'scene_id': item.scene_id,
            'frame_idx': item.frame_idx,
            'cam_type': item.cam_type,
            'filename': str(item.filename),
            'intrinsics': cam_intrinsics_save.flatten().tolist(),
            'extrinsics': cam_extrinsics.flatten().tolist(), # World -> Cam
            'ego_to_world': ego_to_world.flatten().tolist(), # Cam -> World
        })

        # MoGe input info
        H_save, W_save = image_save.shape[:2]
        H_input, W_input = image_input.shape[:2]
        
        fov_x_deg = get_fov_x_deg(W_input, cam_intrinsics_input[0, 0])

        data_dict.update({
            'image_input': torch.from_numpy(image_input).to(torch.float32),
            'fov': fov_x_deg,
        })

        # GT Depth
        data_dict.update({
            'proj_depth': torch.from_numpy(depth_gt).to(torch.float32),
            'image_save': image_save,
            'image_size': (H_input, W_input)
        })

        return data_dict