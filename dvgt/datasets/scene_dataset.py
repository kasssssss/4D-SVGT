import logging
import os.path as osp
import random
import numpy as np
import pandas as pd
from typing import Dict, Literal, Tuple, Optional, List
from torch.utils.data import Dataset
from pathlib import Path

from tools.data_process.utils.io import read_depth, read_image
from dvgt.utils.geometry import (
    to_homogeneous, depth_to_world_coords_points, transform_ego_pose_point3d_to_first_ego,
    transform_point3d_in_world_to_ego_n, transform_extrinsics_to_first_ego, get_relative_future_poses,
    compute_ego_past_to_ego_curr, transform_T_cam_n_ego_first_TO_T_cam_n_cam_first,
    transform_points_in_ego_first_to_cam_first
)
from dvgt.datasets.transforms.transforms import crop_image_depth_and_intrinsic_by_pp, resize_image_depth_and_intrinsic
from dvgt.datasets.transforms.sequence_utils import split_scenes_into_clips, split_scenes_into_windows, \
    sample_training_sequence, get_navtest_logs, get_nuscenes_logs

class DVGTSceneDataset(Dataset):
    """
    Unified Dataset for DVGT.
    Reads pre-processed Parquet metadata and loads images/depths on demand.
    """

    CAM_TYPE_MAP = {
        'nuscene': {
            'CAM_FRONT': 'FRONT', 
            'CAM_FRONT_RIGHT': 'FRONT_RIGHT', 
            'CAM_FRONT_LEFT': 'FRONT_LEFT', 
            'CAM_BACK': 'BACK', 
            'CAM_BACK_RIGHT': 'BACK_RIGHT', 
            'CAM_BACK_LEFT': 'BACK_LEFT'
        },
        'kitti': {
            "2": 'FRONT', 
            "3": 'FRONT_RIGHT',
        },
        'openscene': {
            'CAM_F0': 'FRONT', 
            'CAM_B0': 'BACK', 
            'CAM_L0': 'FRONT_LEFT', 
            'CAM_L1': 'LEFT', 
            'CAM_L2': 'BACK_LEFT', 
            'CAM_R0': 'FRONT_RIGHT', 
            'CAM_R1': 'RIGHT', 
            'CAM_R2': 'BACK_RIGHT'
        },
        'waymo': {
            'FRONT': 'FRONT', 
            'FRONT_LEFT': 'FRONT_LEFT', 
            'FRONT_RIGHT': 'FRONT_RIGHT', 
            'SIDE_LEFT': 'LEFT', 
            'SIDE_RIGHT': 'RIGHT',
        },
        'ddad': {
            'camera_01': 'FRONT', 
            'camera_05': 'FRONT_LEFT', 
            'camera_06': 'FRONT_RIGHT', 
            'camera_07': 'BACK_LEFT', 
            'camera_08': 'BACK_RIGHT', 
            'camera_09': 'BACK'
        },
        "argoverse": {
            "ring_front_center": 'FRONT', 
            "ring_front_left": 'FRONT_LEFT', 
            "ring_side_left": 'LEFT', 
            "ring_rear_left": 'BACK_LEFT', 
            "ring_front_right": 'FRONT_RIGHT', 
            "ring_side_right": 'RIGHT', 
            "ring_rear_right": 'BACK_RIGHT',
        }
    }
    CAM_TYPE_MAP['navtest'] = CAM_TYPE_MAP['openscene']
    CAM_TYPE_MAP['navtrain'] = CAM_TYPE_MAP['openscene']
    CAM_TYPE_MAP['demo'] = CAM_TYPE_MAP['openscene']

    KEYS_TO_SCALE_CONTENT = {
        'points_in_ego_n',
        'points_in_ego_first',
        'points_in_cam_first',
        'ray_depth',
        'depths'
    }

    KEYS_TO_SCALE_TRANSLATION = {
        'ego_n_to_ego_first',
        'ego_first_to_cam_n',
        'cam_first_to_cam_n',
        'future_ego_n_to_ego_curr',
        'ego_past_to_ego_curr',
    }

    def __init__(
        self,
        parquet_path: str = '',
        frame_num_max: int = 99999,   # Adaptive sampling: No hard cap on frames; sequence length is constrained by the total image budget.
        dataset_len: int = -1,      # Set to -1 to use the native sequence duration from the dataset metadata.
        storage_align_depth_path: str = 'data_annotation/align_depth/moge_v2_large_correct_focal',
        storage_proj_depth_path: str = 'data_annotation/proj_depth/moge_v2_large_correct_focal',
        storage_image_path: str = 'data_annotation/image/moge_v2_large_correct_focal',
        image_width: int = 512,
        patch_size: int = 16,
        target_fps: int = 2,
        original_fps: int = 2,
        # Function switch
        enable_depth: bool = False,
        enable_point_in_ego_n: bool = False,
        enable_point_in_ego_first: bool = False,
        enable_ray_depth: bool = False,
        enable_cam: bool = False,
        enable_past_ego: bool = False,
        clip_split_type: str = 'default_train',
        enable_ego_status: bool = False,
        mode: Literal['train', 'val', 'visual']  = 'train',
        filter_lidar: bool = False,
        eval_lidar_proj_depth: bool = False,
        # augmentation params
        aug_pixel_aspect_range: Optional[List[float]] = None, # e.g. [0.9, 1.0]
        aug_safe_scales: Optional[List[float]] = None,  # e.g. [0.8, 1.2]
        rescale: bool = True,
        enable_random_resize: bool = True,
        gt_scale_factor: float = 1.0,
        max_ray_depth: float = 100.0,
        future_frame_num: int = 0,
        inside_random: bool = True,
        # Fixed params for debug and val
        fixed_num_frames: int = 0,
        fixed_num_views: int = 0,
        fixed_aspect_ratio: float = 0.0,
    ):
        """
            view_num_max: Total camera views available in the current dataset.
            frame_num_max: Average frame count used as a sampling heuristic. 
              - Setting this near the average (rather than the absolute maximum) 
              - prevents sampling starvation and avoids infinite retry loops 
              - caused by rare, outlier long-sequences.
        """
        super().__init__()

        # 1. Basic configuration
        self.image_width = image_width
        self.patch_size = patch_size
        self.aug_pixel_aspect_range = aug_pixel_aspect_range
        self.aug_safe_scales = aug_safe_scales
        self.rescale = rescale
        self.enable_random_resize = enable_random_resize
        # Debug/Fixed Settings
        self.fixed_num_frames = fixed_num_frames
        self.fixed_num_views = fixed_num_views
        self.fixed_aspect_ratio = fixed_aspect_ratio

        # 2. Data path configuration
        self.dataset_name = Path(parquet_path).stem.split('_')[0]
        self.full_cam_type = self.CAM_TYPE_MAP[self.dataset_name]
        self.storage_align_depth_path = osp.join(storage_align_depth_path, self.dataset_name)
        self.storage_proj_depth_path = osp.join(storage_proj_depth_path, self.dataset_name)
        self.storage_image_path = osp.join(storage_image_path, self.dataset_name)

        # 3. Function switch
        self.enable_depth = enable_depth
        self.enable_point_in_ego_n = enable_point_in_ego_n
        self.enable_point_in_ego_first = enable_point_in_ego_first
        self.enable_ray_depth = enable_ray_depth
        self.enable_cam = enable_cam
        self.enable_past_ego = enable_past_ego
        self.enable_ego_status = enable_ego_status
        self.mode = mode
        self.filter_lidar = filter_lidar
        self.eval_lidar_proj_depth = eval_lidar_proj_depth
        self.gt_scale_factor = gt_scale_factor
        self.max_ray_depth = max_ray_depth
        self.future_frame_num = future_frame_num

        # 4. Sampling logic configuration
        self.view_num_max = len(self.full_cam_type)
        self.frame_num_max = frame_num_max - self.future_frame_num
        # If True, ignores the input index and samples randomly across current datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = inside_random
        self.fps_step = max(1, original_fps // target_fps)

        # 5. Load Metadata
        logging.info(f"Loading metadata from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        self.scenes: Dict[str, pd.DataFrame] = {
            scene_id: group.set_index(['frame_idx', 'cam_type']).sort_index()
            for scene_id, group in df.groupby('scene_id')
        }

        # eval
        if clip_split_type == 'DVGT1':
            # Divide scenes into non-overlapping clips of length 'self.fixed_num_frames' for inference.
            self.scenes = split_scenes_into_clips(self.scenes, self.fixed_num_frames, self.fps_step, self.dataset_name)
        elif clip_split_type == 'DVGT2':
            # Partition scenes into sliding-window segments; incomplete segments are discarded.
            scene_window = self.fixed_num_frames + self.future_frame_num
            self.scenes = split_scenes_into_windows(self.scenes, scene_window, self.fps_step, self.dataset_name)
        elif clip_split_type == 'navtest':
            self.scenes = get_navtest_logs(
                self.scenes, 
                config_path="third_party/navsim_v1_1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
                num_history_frames=self.fixed_num_frames,
                future_frames=self.future_frame_num
            )
        elif clip_split_type == 'nuscenes_planning':
            self.scenes = get_nuscenes_logs(self.scenes, self.fps_step)
        elif clip_split_type == 'long':
            # Select only the first clip of each scene for the test set.
            self.scenes = split_scenes_into_clips(
                self.scenes, self.fixed_num_frames + self.future_frame_num, 
                self.fps_step, self.dataset_name, only_first_clip=True
            )
        
        # train
        elif clip_split_type == 'default_train':
            pass    # direct use all scene
        elif clip_split_type == 'navtrain':
            self.scenes = get_navtest_logs(
                self.scenes, 
                config_path="third_party/navsim_v1_1/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
            )
        else:
            raise NotImplementedError()

        self.sequence_list = sorted(list(self.scenes.keys()))
        self._dataset_len = dataset_len if dataset_len > 0 else len(self.sequence_list)
        logging.info(f"[{self.mode}]: {self.dataset_name} Real Data size: {len(self.sequence_list)}")
        logging.info(f"[{self.mode}]: {self.dataset_name} Data dataset sample length: {self._dataset_len}\n")

    def __len__(self):
        return self._dataset_len

    def __getitem__(self, idx_info):
        """
        Args:
            idx_info: Tuple (seq_index, views_per_frame, frame_per_seq, aspect_ratio)
                      or just an int for validation.
        """
        if isinstance(idx_info, int):
            seq_index = idx_info
            # Default validation/test params
            views_per_frame = self.fixed_num_views if self.fixed_num_views > 0 else self.view_num_max
            aspect_ratio = self.fixed_aspect_ratio
            num_frames = self.fixed_num_frames
        else:
            seq_index = idx_info[0]
            views_per_frame = self.fixed_num_views if self.fixed_num_views > 0 else idx_info[1]
            num_frames = self.fixed_num_frames if self.fixed_num_frames > 0 else idx_info[2]
            aspect_ratio = self.fixed_aspect_ratio if self.fixed_aspect_ratio > 0 else idx_info[3]

        return self.get_sequence_data(seq_index, num_frames, views_per_frame, aspect_ratio)

    def _calculate_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.

        Args:
            aspect_ratio: Target aspect ratio

        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        image_height = int(self.image_width / aspect_ratio)

        # ensure the input shape is friendly to vision transformer
        if image_height % self.patch_size != 0:
            image_height = (image_height // self.patch_size) * self.patch_size

        image_shape = np.array([image_height, self.image_width])
        return image_shape

    def process_one_image(
        self,
        image: np.ndarray, 
        depth_map: np.ndarray, 
        extrinsics: np.ndarray, 
        intrinsics: np.ndarray, 
        target_shape: np.ndarray,
        track: Optional[np.ndarray] = None,
        filepath: str = ""
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.

        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extrinsics (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intrinsics (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            target_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extrinsics (numpy.ndarray): Updated extrinsic matrix,
                intrinsics (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information
            )
        """
        original_size = np.array(image.shape[:2])
        aug_size = original_size

        # Apply random scale augmentation during training
        if self.aug_pixel_aspect_range is not None:
            rand_h, rand_w = np.random.uniform(self.aug_pixel_aspect_range[0], self.aug_pixel_aspect_range[1], 2)
            aug_ratio = (rand_w / rand_h) * (intrinsics[0, 0] / intrinsics[1, 1])
            if self.aug_safe_scales and (aug_ratio < self.aug_safe_scales[0] or aug_ratio > self.aug_safe_scales[1]):
                rand_h, rand_w = 1.0, 1.0
            aug_size = (original_size * np.array([rand_h, rand_w])).astype(np.int32)

        # Move principal point to the image center and crop if necessary
        image, depth_map, intrinsics, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intrinsics, aug_size, track=track, filepath=filepath,
        )

        # Resize images and update intrinsics
        if self.rescale:
            image, depth_map, intrinsics, track = resize_image_depth_and_intrinsic(
                image, depth_map, intrinsics, target_shape, track=track,
                enable_random_resize=self.enable_random_resize    
            )
        else:
            print("Not rescaling the images")

        # Ensure final crop to target shape
        image, depth_map, intrinsics, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intrinsics, target_shape, track=track, filepath=filepath, strict=True,
        )

        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
            depth_map, extrinsics, intrinsics
        )

        return (
            image,
            depth_map,
            extrinsics,
            intrinsics,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

    def load_one_image(
        self,
        item_info: pd.Series, 
        target_shape: np.ndarray
    ) -> Tuple:
        filename = item_info['filename']
        image_filepath = osp.join(self.storage_image_path, f"{filename}.jpg")
        image = read_image(image_filepath)

        is_lidar_mode = item_info.get('use_lidar_proj_depth', False) and self.mode != 'visual'

        if is_lidar_mode and self.filter_lidar:
            depth_map = np.zeros(image.shape[:2], dtype=np.float32) 
        else:
            if is_lidar_mode or self.eval_lidar_proj_depth:
                depth_dir = self.storage_proj_depth_path
            else:
                depth_dir = self.storage_align_depth_path
            
            depth_filepath = osp.join(depth_dir, f"{filename}.png")
            depth_map, _ = read_depth(depth_filepath)

        depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
        depth_map[depth_map < 1] = 0    # Security assurance

        # We adopt the OpenCV (RDF) convention for both camera extrinsics and ego-to-world transforms.
        # Since initial loading is in float64, high precision is preserved for all downstream point cloud operations.
        intrinsics = item_info['intrinsics'].reshape(3, 3)    
        extrinsics = item_info['extrinsics'].reshape(3, 4)    

        return self.process_one_image(      # Ensure world points are computed in float64 for numerical safety.
            image, depth_map, extrinsics, intrinsics, target_shape, filepath=image_filepath
        )

    def get_sequence_data(
        self,
        seq_index: int,
        num_frames: int, 
        views_per_frame: int, 
        aspect_ratio: float = 1.0
    ) -> Dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            num_frames (int): Number of frame per sequence.
            views_per_frame (int): Number of images per frame.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, point3d, and other metadata.
        """
        if self.mode == 'train':
            seq_name, scene_df, sampled_frame_ids, future_frame_ids = sample_training_sequence(
                sequence_list=self.sequence_list, scenes=self.scenes, fps_step=self.fps_step,
                num_frames=num_frames, seq_index=seq_index, future_frame_num=self.future_frame_num,
                inside_random=self.inside_random
            )
            if self.fixed_num_views > 0:
                # FIXME: The number of views is currently hard-coded for either debugging or full-view fine-tuning.
                selected_cams = list(self.full_cam_type.keys())
            else:
                selected_cams = random.sample(list(self.full_cam_type.keys()), views_per_frame)
        else:
            seq_name = self.sequence_list[seq_index]
            scene_df = self.scenes[seq_name]
            total_frame_ids = scene_df.index.get_level_values('frame_idx').unique().sort_values()
            if self.future_frame_num > 0:
                sampled_frame_ids = total_frame_ids[:-self.future_frame_num]
                future_frame_ids = total_frame_ids[-self.future_frame_num:]
            else:
                sampled_frame_ids = total_frame_ids
                future_frame_ids = []
            selected_cams = list(self.full_cam_type.keys())
            
        target_shape = self._calculate_target_shape(aspect_ratio)
        H, W = target_shape
        T, V = len(sampled_frame_ids), len(selected_cams)


        # Pre-allocate arrays to reduce memory overhead. 
        # Use float64 for all 'world' coordinates to avoid numerical degradation. (world point, ego2world, cam2world)
        batch_data = {
            'seq_name': seq_name,
            'images': np.zeros((T, V, H, W, 3), dtype=np.uint8),
            'points': np.zeros((T, V, H, W, 3), dtype=np.float64),
            'point_masks': np.zeros((T, V, H, W), dtype=bool),
            'ego_to_worlds': np.zeros((T, 3, 4), dtype=np.float64),
            'cam_types': [self.full_cam_type[cam] for cam in selected_cams],
            'intrinsics': np.zeros((T, V, 3, 3), dtype=np.float32),
            'extrinsics': np.zeros((T, V, 3, 4), dtype=np.float64),
        }
        if self.enable_depth:
            batch_data['depths'] = np.zeros((T, V, H, W), dtype=np.float32)
        
        if self.enable_ego_status:
            # dim=8：vel, acc, command
            batch_data['ego_status'] = np.zeros((T, 8), dtype=np.float32)

        for t, frame_id in enumerate(sampled_frame_ids):
            # Extract ego pose from the first view (ego pose is identical across all cameras for the same frame).
            ego_pose = scene_df.loc[(frame_id, selected_cams[0]), 'ego_to_world'].reshape(3, 4)
            batch_data['ego_to_worlds'][t] = ego_pose

            if self.enable_ego_status:
                vel = scene_df.loc[(frame_id, selected_cams[0]), 'ego_velocity']
                acc = scene_df.loc[(frame_id, selected_cams[0]), 'ego_acceleration']
                command = scene_df.loc[(frame_id, selected_cams[0]), 'driving_command']
                batch_data['ego_status'][t] = np.concatenate([vel, acc, command])

            for v, cam_type in enumerate(selected_cams):
                item_info = scene_df.loc[(frame_id, cam_type)]
                (
                    image, depth, extrinsics, intrinsics,
                    world_coords_points, cam_coords_points, point_mask, track
                ) = self.load_one_image(item_info, target_shape)

                batch_data['images'][t, v] = image
                batch_data['points'][t, v] = world_coords_points
                batch_data['point_masks'][t, v] = point_mask
                batch_data['intrinsics'][t, v] = intrinsics
                batch_data['extrinsics'][t, v] = extrinsics
                if self.enable_depth:
                    batch_data['depths'][t, v] = depth

        ego_to_worlds_homo = to_homogeneous(batch_data['ego_to_worlds'])
        extrinsics_homo = to_homogeneous(batch_data['extrinsics'])

        # Eliminate world-frame dependency to prevent downstream errors by normalizing all poses relative to the first-frame's ego-coordinate.
        batch_data['ego_n_to_ego_first'], points_in_ego_first = transform_ego_pose_point3d_to_first_ego(
            batch_data['points'], ego_to_worlds_homo
        )
        batch_data['ego_first_to_cam_n'] = transform_extrinsics_to_first_ego(extrinsics_homo, ego_to_worlds_homo)
        del batch_data['ego_to_worlds'], batch_data['extrinsics'], batch_data['points']

        if self.enable_cam:
            batch_data['cam_first_to_cam_n'] = transform_T_cam_n_ego_first_TO_T_cam_n_cam_first(batch_data['ego_first_to_cam_n'])
            batch_data['points_in_cam_first'] = transform_points_in_ego_first_to_cam_first(points_in_ego_first, batch_data['ego_first_to_cam_n'])

        points_in_ego_n = transform_point3d_in_world_to_ego_n(
            world_points=points_in_ego_first,
            ego_to_worlds=batch_data['ego_n_to_ego_first']
        )

        if self.enable_point_in_ego_n:
            batch_data['points_in_ego_n'] = points_in_ego_n
        
        if self.enable_point_in_ego_first:
            batch_data['points_in_ego_first'] =  points_in_ego_first        

        # Obtain ray depth in the ego-coordinate system for each frame and clip the point cloud.
        ray_depth = np.linalg.norm(points_in_ego_n, axis=-1)
        batch_data['point_masks'] = batch_data['point_masks'] & (ray_depth < self.max_ray_depth)

        if self.enable_ray_depth:
            ray_direction = points_in_ego_n / (ray_depth[..., None] + 1e-6)
            batch_data['ray_depth'] = ray_depth
            batch_data['ray_direction'] = ray_direction

        if self.future_frame_num > 0 and len(future_frame_ids) > 0:
            future_poses = np.zeros((self.future_frame_num, 3, 4), dtype=np.float64)
            for i, frame_id in enumerate(future_frame_ids):
                future_poses[i] = scene_df.loc[(frame_id, selected_cams[0]), 'ego_to_world'].reshape(3, 4)
            future_poses = to_homogeneous(future_poses)
            # NOTE Calculating relative pose. Note that dependency on the world coordinate system remains inherent and cannot be decoupled.
            batch_data['future_ego_n_to_ego_curr'] = get_relative_future_poses(ego_to_worlds_homo, future_poses)
        
        if self.enable_past_ego:
            batch_data['ego_past_to_ego_curr'] = compute_ego_past_to_ego_curr(batch_data['ego_n_to_ego_first'])

        if self.gt_scale_factor != 1.0:
            for key in batch_data.keys():
                if key in self.KEYS_TO_SCALE_CONTENT:
                    batch_data[key] *= self.gt_scale_factor
                elif key in self.KEYS_TO_SCALE_TRANSLATION:
                    batch_data[key][..., :3, 3] *= self.gt_scale_factor
                elif key == 'ego_status':
                    batch_data[key][..., :4] *= self.gt_scale_factor
        return batch_data