import torch
import time
import logging
import os
from typing import Dict, Tuple
from einops import rearrange
import sys
import numpy as np
import torchvision.transforms as tvf
repo_path = "third_party/map-anything"

# Pre-flight Check: Ensure the source code is available.
if not os.path.exists(repo_path):
    raise FileNotFoundError(
        f"repo dependency missing at: {repo_path}\n"
        "Action required: cd third_party; git clone https://github.com/facebookresearch/map-anything.git"
    )

sys.path.insert(0, os.path.abspath(repo_path))

from mapanything.models import MapAnything
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from mapanything.utils.cropping import crop_resize_if_necessary

from dvgt.evaluation.utils.geometry import convert_camera_0_pose_to_ego_0_pose, convert_camera_0_point_to_ego_0_point
from dvgt.utils.geometry import closed_form_inverse_se3
from .base_model import BaseEvaluatorModel

def load_images_from_tensor(
    images_tensor: torch.Tensor,
    resize_mode="fixed_size",
    size=None,
    norm_type="dinov2",
    patch_size=14,
    verbose=False,
):
    """
    Open and convert all images in a list or folder to proper input format for model

    Args:
        images_tensor: [N, 3, H, W]
        resize_mode (str): Resize mode - "fixed_mapping", "longest_side", "square", or "fixed_size". Defaults to "fixed_mapping".
        size (int or tuple, optional): For "fixed_size": tuple of (width, height)
        norm_type (str, optional): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys. Defaults to "dinov2".
        patch_size (int, optional): Patch size for image processing. Defaults to 14.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Returns:
        list: List of dictionaries containing image data and metadata
    """
    # Validate resize_mode and size parameter requirements
    valid_resize_modes = ["fixed_mapping", "longest_side", "square", "fixed_size"]
    if resize_mode not in valid_resize_modes:
        raise ValueError(
            f"Resize_mode must be one of {valid_resize_modes}, got '{resize_mode}'"
        )

    # if resize_mode in ["longest_side", "square", "fixed_size"] and size is None:
    #     raise ValueError(f"Size parameter is required for resize_mode='{resize_mode}'")

    # Validate size type based on resize mode
    if resize_mode in ["longest_side", "square"]:
        if not isinstance(size, int):
            raise ValueError(
                f"Size must be an int for resize_mode='{resize_mode}', got {type(size)}"
            )
    elif resize_mode == "fixed_size":
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(
                f"Size must be a tuple/list of (width, height) for resize_mode='fixed_size', got {size}"
            )
        if not all(isinstance(x, int) for x in size):
            raise ValueError(
                f"Size values must be integers for resize_mode='fixed_size', got {size}"
            )

    # First pass: Load all images and collect aspect ratios
    loaded_images = []
    aspect_ratios = []
    images_np = images_tensor.cpu().numpy()
    for img in images_np:
        loaded_images.append(img)
        H1, W1, _ = img.shape
        aspect_ratios.append(W1 / H1)

    # Calculate average aspect ratio and determine target size
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    if verbose:
        print(
            f"Calculated average aspect ratio: {average_aspect_ratio:.3f} from {len(aspect_ratios)} images"
        )

    # Determine target size for all images based on resize mode
    if resize_mode == "fixed_size":
        # Use exact size provided, aligned to patch_size
        target_size = (
            (size[0] // patch_size) * patch_size,
            (size[1] // patch_size) * patch_size,
        )

    if verbose:
        print(
            f"Using target resolution {target_size[0]}x{target_size[1]} (W x H) for all images"
        )

    # Get the image normalization function based on the norm_type
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    # Second pass: Resize all images to the same target size
    imgs = []
    for img in loaded_images:
        # Resize and crop the image to the target size
        img = crop_resize_if_necessary(img, resolution=target_size)[0]

        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
                data_norm_type=[norm_type],
            )
        )

    assert imgs, "No images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")

    return imgs

class MapAnythingWrapper(BaseEvaluatorModel):
    def load(self, checkpoint_path: str):
        self.model = MapAnything.from_pretrained(checkpoint_path)

    def model_infer(self, images):
        B, T, V, H, W, _ = images.shape
        
        images = images.flatten(1, 2)   # [1, T*V, H, W, 3]

        processed_views = load_images_from_tensor(images[0], size=(W, H), resize_mode="fixed_size")

        predictions = self.model.infer(
            processed_views,                            # Input views
            memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
            use_amp=True,                     # Use mixed precision inference (recommended)
            amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
            apply_mask=False,                  # Apply masking to dense geometry outputs
            mask_edges=False,                  # Remove edge artifacts by using normals and depth
            apply_confidence_mask=False,      # Filter low-confidence regions
        )

        return predictions

    def infer(self, batch) -> Tuple[Dict, Dict]:
        # images: [1, T, V, C, H, W]
        images = rearrange(batch['images'], 'b t v c h w -> b t v h w c')
        images = (images * 255).to(torch.uint8)

        B, T, V, H, W, _ = images.shape
        assert B == 1, "Batch size must be 1"        
        start_time = time.time()

        predictions = self.model_infer(images)

        model_end_time = time.time()
        logging.debug(f"Model Forward Pass: {model_end_time - start_time:.4f}s")

        pred_points_list = []
        camera_poses_list = []
        for i, pred in enumerate(predictions):
            pts3d = pred["pts3d"]                     # 3D points in world coordinates (B, H, W, 3)
            camera_poses = pred["camera_poses"]       # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)

            pred_points_list.append(pts3d)
            camera_poses_list.append(camera_poses)

        pred_points_in_world = torch.stack(pred_points_list, dim=1)    # B, N, H, W, 3
        camn_to_cam0 = torch.stack(camera_poses_list, dim=1)    # B, N, 4, 4
        cam0_to_camn = closed_form_inverse_se3(camn_to_cam0)

        pred_points = pred_points_in_world.reshape(B, T, V, H, W, 3)
        cam0_to_camn = cam0_to_camn.reshape(B, T, V, 4, 4)

        outputs = {}

        # process pose
        outputs['ego_n_to_ego_first'] = convert_camera_0_pose_to_ego_0_pose(
            pred_T_cam_n_cam_0=cam0_to_camn[:, :, 0],
            batch=batch,
            use_umeyama_per_scene=False
        )

        # process points
        pred_ray_depth, pred_points_in_ego_first = convert_camera_0_point_to_ego_0_point(
            pred_points_in_cam0=pred_points,
            batch=batch,
            use_umeyama_per_scene=False
        )

        outputs['points_in_ego_first'] = pred_points_in_ego_first
        outputs['ray_depth'] = pred_ray_depth

        return batch, outputs
