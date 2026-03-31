import torch
import time
import logging
import os
from typing import Dict, Tuple
import numpy as np
from copy import deepcopy
from third_party.CUT3R.src.dust3r.model import ARCroco3DStereo

from dvgt.evaluation.utils.geometry import convert_camera_0_pose_to_ego_0_pose, convert_camera_0_point_to_ego_0_point
from dvgt.utils.geometry import closed_form_inverse_se3
from dvgt.utils.pose_encoding import decode_pose
from .base_model import BaseEvaluatorModel

def prepare_input_tensor(
    img_tensor, img_mask, raymaps=None, raymap_mask=None, revisit=1, update=True
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_tensor torch.Tensor: [N, 3, H, W] in cpu
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    images = []
    for idx, img in enumerate(img_tensor):
        images.append({
            "img": ((img - 0.5) / 0.5)[None], 
            "true_shape": np.array(img.shape[-2:]), 
            "idx": idx, 
            "instance": str(idx)
        })
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views

class CUT3RWrapper(BaseEvaluatorModel):
    def load(self, checkpoint_path: str):
        self.model = ARCroco3DStereo.from_pretrained(checkpoint_path)

    def infer(self, batch) -> Tuple[Dict, Dict]:
        # images: [1, T, V, C, H, W]
        images = batch['images']
        B, T, V, _, H, W = images.shape
        device = images.device
        images = images.flatten(1, 2)   # [1, T*V, C, H, W]
        img_mask = [True] * T

        views = prepare_input_tensor(
            img_tensor=images[0],
            img_mask=img_mask,
            revisit=1,
            update=True,
        )
        ignore_keys = set(
            ["img", "depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
        )

        for view in views:
            for name in view.keys():  # pseudo_focal
                if name in ignore_keys:
                    continue
                if isinstance(view[name], tuple) or isinstance(view[name], list):
                    view[name] = [x.to(device, non_blocking=True) for x in view[name]]
                else:
                    view[name] = view[name].to(device, non_blocking=True)

        start_time = time.time()
        with torch.amp.autocast(device_type='cuda', enabled=False):
            output = self.model(views)
            preds = output.ress
        model_end_time = time.time()
        logging.debug(f"Model Forward Pass: {model_end_time - start_time:.4f}s")

        with torch.amp.autocast(device_type='cuda', enabled=False):
            pred_points_in_cam0 = []
            pred_T_cam_0_cam_n = []

            for pred in preds:
                pred_points_in_cam0.append(pred['pts3d_in_other_view'])
                pred_T_cam_0_cam_n.append(pred['camera_pose'])
            
            pred_points_in_cam0 = torch.cat(pred_points_in_cam0, dim=0).reshape(B, T, V, H, W, 3)
            pred_T_cam_0_cam_n = torch.cat(pred_T_cam_0_cam_n, dim=0).reshape(B, T, V, 7)
            
            pred_T_cam_0_cam_n, _ = decode_pose(pred_T_cam_0_cam_n, pose_encoding_type="absT_quaR")
            pred_T_cam_n_cam_0 = closed_form_inverse_se3(pred_T_cam_0_cam_n)
        
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

        return batch, outputs
