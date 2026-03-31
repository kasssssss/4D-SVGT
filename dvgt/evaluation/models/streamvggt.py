import torch
import time
import logging
import os
from einops import rearrange
from typing import Dict, Tuple
import sys

repo_path = "third_party/StreamVGGT/src"

# Pre-flight Check: Ensure the source code is available.
if not os.path.exists(repo_path):
    raise FileNotFoundError(
        f"repo dependency missing at: {repo_path}\n"
        "Action required: cd third_party; git clone https://github.com/wzzheng/StreamVGGT.git"
    )

sys.path.insert(0, os.path.abspath(repo_path))

from streamvggt.models.streamvggt import StreamVGGT
from dvgt.evaluation.utils.geometry import convert_camera_0_pose_to_ego_0_pose, convert_camera_0_point_to_ego_0_point
from dvgt.utils.pose_encoding import decode_pose
from .base_model import BaseEvaluatorModel

class StreamVGGTWrapper(BaseEvaluatorModel):
    def load(self, checkpoint_path: str):
        self.model = StreamVGGT()
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=True)

    def infer(self, batch) -> Tuple[Dict, Dict]:
        B, T, V, _, H, W = batch['images'].shape
        assert B == 1
        images = rearrange(batch['images'], 'b t v c h w -> (b t v) c h w')

        frames = []
        for i in range(images.shape[0]):
            image = images[i].unsqueeze(0) 
            frame = {
                "img": image
            }
            frames.append(frame)

        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = self.model.inference(frames)
        model_end_time = time.time()
        logging.debug(f"Model Forward Pass: {model_end_time - start_time:.4f}s")

        all_pts3d = []
        all_conf = []
        all_camera_pose = []
        
        for res in output.ress:
            all_pts3d.append(res['pts3d_in_other_view'].squeeze(0))
            all_conf.append(res['conf'].squeeze(0))
            all_camera_pose.append(res['camera_pose'].squeeze(0))

        predictions = {}    
        predictions["world_points"] = torch.stack(all_pts3d, dim=0)  # (S, H, W, 3)
        predictions["world_points_conf"] = torch.stack(all_conf, dim=0)  # (S, H, W)
        predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0)  # (S, 9)
        extrinsic, _ = decode_pose(
            predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"], 
            images.shape[-2:], pose_encoding_type="absT_quaR_FoV"
        )

        pred_points_in_cam0 = predictions["world_points"].reshape(B, T, V, H, W, 3)
        pred_T_cam_n_cam_0 = extrinsic.reshape(B, T, V, 4, 4)

        outputs = {}

        # process pose
        outputs['ego_n_to_ego_first'] = convert_camera_0_pose_to_ego_0_pose(
            pred_T_cam_n_cam_0=pred_T_cam_n_cam_0[:, :, 0],
            batch=batch,
            use_umeyama_per_scene=True
        )

        # process points
        pred_ray_depth, pred_points_in_ego_first = convert_camera_0_point_to_ego_0_point(
            pred_points_in_cam0=pred_points_in_cam0,
            batch=batch,
            use_umeyama_per_scene=True
        )

        outputs['points_in_ego_first'] = pred_points_in_ego_first
        outputs['ray_depth'] = pred_ray_depth
        outputs['points_in_ego_first_conf'] = predictions['world_points_conf']

        return batch, outputs
