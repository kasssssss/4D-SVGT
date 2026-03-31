from dataclasses import dataclass
import cv2
import re
from typing import Tuple, List, Dict, Optional
import pykitti
import numpy as np
from pykitti.raw import raw
import torch
import logging
from pathlib import Path
from functools import lru_cache

from .factory import register_dataset
from .base_dataset import BaseDataset
from dvgt.utils.geometry import closed_form_inverse_se3
from tools.data_process.utils.transformer_np import project_lidar_to_depth
from third_party.MoGe.moge.utils.geometry_torch import mask_aware_nearest_resize
from tools.data_process.utils.tools import gen_and_create_output_dirs
from tools.data_process.utils.image import process_image_sequentially, resize_to_aspect_ratio, get_fov_x_deg

logger = logging.getLogger(__name__)

@register_dataset('kitti')
class KittiDataset(BaseDataset):
    CAMERA_NAMES = [2, 3]
    FRAME_TRIM_COUNT = 5  # 删除前后各5帧
    PROJ_DEPTH_SCALE_FACTOR = 256.0
    VAL_SCENE_NAMES = {     # kitti depth prediction官方划分的验证集
        '2011_09_26_drive_0002_sync', '2011_09_26_drive_0005_sync', '2011_09_26_drive_0013_sync', '2011_09_26_drive_0020_sync', '2011_09_26_drive_0023_sync', 
        '2011_09_26_drive_0036_sync', '2011_09_26_drive_0079_sync', '2011_09_26_drive_0095_sync', '2011_10_03_drive_0047_sync', '2011_09_30_drive_0016_sync', 
        '2011_09_26_drive_0113_sync', '2011_09_28_drive_0037_sync', '2011_09_29_drive_0026_sync', 
    }
    EXCLUDED_SCENES = {
        # 无depth gt
        '2011_10_03_drive_0058_sync',   
        '2011_09_28_drive_0225_sync',   # 缺失image02
        '2011_09_29_drive_0108_sync'    # 缺失image03

        # gt pose错误, TODO 这里删的太多了，或许可以少删一些
        '2011_09_26_drive_0093_sync',
        '2011_09_30_drive_0028_sync',     # 865-1075帧，10hz
    }

    def __init__(
        self,
        data_root: str = 'public_datasets/kitti',
        storage_pred_depth_path: str = 'public_datasets/data_annotation/pred_depth/moge_v2_large/kitti',
        storage_align_depth_path: str = 'public_datasets/data_annotation/align_depth/moge_v2_large_correct_focal/kitti',
        storage_proj_depth_path: str = 'public_datasets/data_annotation/proj_depth/moge_v2_large_correct_focal/kitti',
        storage_image_path: str = 'public_datasets/data_annotation/image/moge_v2_large_correct_focal/kitti',
        num_tokens: int = 2500,
        official_proj_depth_root: str = "public_datasets/kitti/official_depth_annotation",
        use_official_proj_depth: bool = True,
        split: str = 'train',    # 'train' or 'val'
    ) -> None:
        # 配置gt depth
        self.official_proj_depth_root = Path(official_proj_depth_root)
        self.use_official_proj_depth = use_official_proj_depth
        self.split = split

        super().__init__(data_root, storage_pred_depth_path, storage_proj_depth_path, storage_image_path, storage_align_depth_path, num_tokens)

    @dataclass(frozen=True, slots=True)
    class SampleData:
        scene_id: str
        date: str
        drive: str
        cam_type: str
        frame_idx: int
        image_path: Path
        filename: Path

    def _gather_samples(self) -> List[SampleData]:
        drive_sync_dirs = self.data_root.glob("2011_*_*/2011_*_drive_*_sync")
        
        if self.split == 'train':
            drive_sync_dirs = [d for d in drive_sync_dirs if d.name not in self.VAL_SCENE_NAMES]
        else:
            drive_sync_dirs = [d for d in drive_sync_dirs if d.name in self.VAL_SCENE_NAMES]

        samples_to_process = []
        for drive_dir in drive_sync_dirs:
            # 2011_09_28/2011_09_28_drive_0001_sync
            if drive_dir.name in self.EXCLUDED_SCENES:
                continue
            date = drive_dir.parent.name
            drive = re.search(r'drive_(\d{4})_sync', drive_dir.name).group(1)
            for cam_type in self.CAMERA_NAMES:
                image_paths = sorted(drive_dir.glob(f"image_{cam_type:02d}/data/*.png"))
                image_paths = image_paths[self.FRAME_TRIM_COUNT: -self.FRAME_TRIM_COUNT]     # 这里删除前五帧和后五帧数据，因为没有depth proj gt标注
                for img_path in image_paths:
                    frame_idx = int(img_path.stem)
                    relative_image_path = img_path.relative_to(self.data_root).with_suffix('')
                    samples_to_process.append(self.SampleData(
                        scene_id=drive_dir.name,
                        cam_type=str(cam_type),
                        frame_idx=frame_idx,
                        image_path=img_path,
                        filename=relative_image_path,
                        date=date,
                        drive=drive
                    ))
        logging.info(f"处理：{len(samples_to_process)}条数据")

        return samples_to_process
    
    # 懒加载加载pykitti handler
    @lru_cache(maxsize=16)
    def _get_handler(self, date: str, drive: str) -> raw:
        return pykitti.raw(self.data_root, date, drive)

    def _get_image_and_meta_info(self, handler: raw, item: SampleData) -> Tuple:
        image = cv2.cvtColor(cv2.imread(str(item.image_path)), cv2.COLOR_BGR2RGB)
        original_aspect_ratio = image.shape[1] / image.shape[0]
        
        cam_intrinsics = getattr(handler.calib, f'K_cam{item.cam_type}')

        assert np.isclose(cam_intrinsics[0, 0], cam_intrinsics[1, 1]), f"kitti fx != fy: {item.image_path}"

        image_input, cam_intrinsics_input, image_crop_indice = process_image_sequentially(image, cam_intrinsics, num_tokens=self.num_tokens)

        image_save, cam_intrinsics_save = resize_to_aspect_ratio(image_input, cam_intrinsics_input, original_aspect_ratio)

        # 外参
        T_w_imu = handler.oxts[item.frame_idx].T_w_imu
        T_imu_w = closed_form_inverse_se3(T_w_imu)
        T_cam_imu = getattr(handler.calib, f'T_cam{item.cam_type}_imu')
        cam_extrinsics = T_cam_imu @ T_imu_w
        T_world_ego_rdf = T_w_imu @ self.T_flu_rdf

        # 速度加速度
        ego_dynamic_state = handler.oxts[item.frame_idx].packet
        ego_velocity_rdf = np.array((-ego_dynamic_state.vl, ego_dynamic_state.vf), dtype=np.float32)
        ego_acceleration_rdf = np.array((-ego_dynamic_state.al, ego_dynamic_state.af), dtype=np.float32)
        driving_command = np.array([0, 0, 0, 1], dtype=np.float32)
        
        return (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, cam_extrinsics[:3], T_world_ego_rdf[:3], 
            image_crop_indice,
            ego_velocity_rdf, ego_acceleration_rdf, driving_command
        )

    @staticmethod
    def _gen_lidar_proj_depth(handler: raw, item: SampleData, cam_intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
        # 生成lidar proj depth
        lidar_pc = handler.get_velo(item.frame_idx)[:, :3]
        lidar_to_cam = getattr(handler.calib, f'T_cam{item.cam_type}_velo')

        proj_depth = project_lidar_to_depth(
            points=lidar_pc,
            points_coordinate_to_camera=lidar_to_cam,
            camera_intrinsic=cam_intrinsics,
            height=height, width=width
        )
        proj_depth = torch.from_numpy(proj_depth)
        return proj_depth

    def _try_load_official_depth(self, item: SampleData, image_crop_indice: Tuple[slice, slice], height: int, width: int) -> Optional[torch.Tensor]:
        """
            直接读取官方depth gt
            official depth gt，有些文件没有，使用proj depth代替
            比如：
                '2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000177.png',
                '2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000178.png',
                '2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000179.png',
                '2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000180.png',
                '2011_09_26/2011_09_26_drive_0009_sync/image_03/data/0000000177.png',
                '2011_09_26/2011_09_26_drive_0009_sync/image_03/data/0000000178.png',
                '2011_09_26/2011_09_26_drive_0009_sync/image_03/data/0000000179.png',
                '2011_09_26/2011_09_26_drive_0009_sync/image_03/data/0000000180.png',
        """
        proj_depth_path = self.official_proj_depth_root / self.split / item.scene_id / "proj_depth" / "groundtruth" / f"image_{int(item.cam_type):02d}" / f"{item.frame_idx:010d}.png"
        
        if not proj_depth_path.is_file():
            logging.warning(f"Official projected depth not found, will generate from LiDAR: {proj_depth_path}")
            return None
        
        proj_depth = cv2.imread(str(proj_depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / self.PROJ_DEPTH_SCALE_FACTOR
        proj_depth = proj_depth[image_crop_indice]
        proj_depth, proj_depth_mask = mask_aware_nearest_resize(torch.from_numpy(proj_depth), torch.from_numpy(proj_depth != 0), (width, height))
        proj_depth = torch.where(proj_depth_mask, proj_depth, torch.inf)
        return proj_depth

    def _get_proj_depth(self, item: SampleData, handler: raw, cam_intrinsics: np.ndarray, height: int, width: int, image_crop_indice: Tuple[slice, slice]) -> torch.Tensor:
        proj_depth = None
        
        if self.use_official_proj_depth:
            proj_depth = self._try_load_official_depth(item, image_crop_indice, height, width)

        if proj_depth is None:
            proj_depth = self._gen_lidar_proj_depth(handler, item, cam_intrinsics, height, width)
            
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

        handler = self._get_handler(item.date, item.drive)

        (
            cam_intrinsics_input, image_input, 
            cam_intrinsics_save, image_save, cam_extrinsics, ego_to_world, 
            image_crop_indice,
            ego_velocity_rdf, ego_acceleration_rdf, driving_command
        ) = self._get_image_and_meta_info(handler, item)

        data_dict.update({
            'scene_id': item.scene_id,
            'frame_idx': item.frame_idx,  # kitti的idx从5开始
            'cam_type': item.cam_type,
            'filename': str(item.filename),
            'intrinsics': cam_intrinsics_save.flatten().tolist(),
            'extrinsics': cam_extrinsics.flatten().tolist(),
            'ego_to_world': ego_to_world.flatten().tolist(),
            'ego_velocity': ego_velocity_rdf.tolist(),
            'ego_acceleration': ego_acceleration_rdf.tolist(),
            'driving_command': driving_command.tolist(),
        })

        H_save, W_save = image_save.shape[:2]
        H_input, W_input = image_input.shape[:2]
        
        fov_x_deg = get_fov_x_deg(W_input, cam_intrinsics_input[0, 0])

        # MoGe input
        data_dict.update({
            'image_input': torch.from_numpy(image_input).to(torch.float32),
            'fov': fov_x_deg,    # float
        })
        
        proj_depth = self._get_proj_depth(item, handler, cam_intrinsics_save,  H_save, W_save, image_crop_indice)
        
        data_dict.update({
            'proj_depth': proj_depth.to(torch.float32),
            'image_save': image_save,     # 后续可以直接保存这个 uint8 的图片
            'image_size': (H_input, W_input),
        })

        return data_dict