# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from typing import Dict
from safetensors.torch import load_file
from iopath.common.file_io import g_pathmgr
import time
import logging

from third_party.vggt.models.aggregator import Aggregator
from third_party.vggt.heads.camera_head import CameraHead
from third_party.vggt.heads.dpt_head import DPTHead
from third_party.vggt.heads.track_head import TrackHead
from dvgt.utils.pose_encoding import decode_pose
from dvgt.evaluation.utils.geometry import convert_camera_0_pose_to_ego_0_pose, convert_camera_0_point_to_ego_0_point

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # In fine-tuning, Images shape is [B, T, V, 3, H, W]
        # If without batch dimension, add it
        if len(images.shape) == 5:
            images = images.unsqueeze(0)
        B, T, V, _, H, W = images.shape
        images = images.flatten(1, 2)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            if self.camera_head is not None:
                model_start_time = time.time()
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                model_end_time = time.time()
                logging.debug(f"ego pose head: {(model_end_time - model_start_time):.4f}s")

                pose_enc_list = [pose_enc.reshape(B, T, V, -1) for pose_enc in pose_enc_list]
                predictions["cam_pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["cam_pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, frames_chunk_size=None
                )
                predictions["depth"] = depth.reshape(B, T, V, H, W, 1)
                predictions["depth_conf"] = depth_conf.reshape(B, T, V, H, W)

            if self.point_head is not None:
                model_start_time = time.time()
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, frames_chunk_size=None
                )
                model_end_time = time.time()
                logging.debug(f"point head: {(model_end_time - model_start_time):.4f}s")
                
                predictions["points"] = pts3d.reshape(B, T, V, H, W, 3)
                predictions["points_conf"] = pts3d_conf.reshape(B, T, V, H, W)

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images.reshape(B, T, V, 3, H, W)  # store the images for visualization during inference

        return predictions

    def custom_load_state_dict(self, checkpoint_conf: Dict):
        if checkpoint_conf.get('direct_load_pretrained_weights_path', None):
            state_dict = load_file(checkpoint_conf['direct_load_pretrained_weights_path'], device="cpu")
    
            missing, unexpected = self.load_state_dict(
                state_dict, strict=checkpoint_conf.get('strict', True)
            )
            return missing, unexpected, None
        elif checkpoint_conf.get('resume_checkpoint_path', None):
            ckpt_path = checkpoint_conf['resume_checkpoint_path']

            with g_pathmgr.open(ckpt_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            
            # Load model state
            model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            missing, unexpected = self.load_state_dict(
                model_state_dict, strict=True
            )
            return missing, unexpected, checkpoint
        else:
            raise NotImplementedError("No pretrained weights path provided for loading.")

    @staticmethod
    def post_process_for_eval(
        predictions: Dict, 
        batch: Dict, 
        **kwargs,
    ) -> Dict:
        outputs = {}
        # process pose
        # Deriving ego-pose from Front Camera (Index 0) via extrinsic mapping.
        pred_pose_enc = predictions['cam_pose_enc'][:, :, 0]
        pred_extrinsic, _ = decode_pose(pred_pose_enc, pose_encoding_type="absT_quaR_FoV", build_intrinsics=False)

        outputs['ego_n_to_ego_first'] = convert_camera_0_pose_to_ego_0_pose(
            pred_T_cam_n_cam_0=pred_extrinsic,
            batch=batch,
            use_umeyama_per_scene=kwargs.get('use_umeyama_per_scene', True)
        )

        # process points
        pred_ray_depth, pred_points_in_ego_first = convert_camera_0_point_to_ego_0_point(
            pred_points_in_cam0=predictions['points'],
            batch=batch,
            use_umeyama_per_scene=kwargs.get('use_umeyama_per_scene', True)
        )

        outputs['points_in_ego_first'] = pred_points_in_ego_first
        outputs['points_in_ego_first_conf'] = predictions['points_conf']
        outputs['ray_depth'] = pred_ray_depth

        return outputs
