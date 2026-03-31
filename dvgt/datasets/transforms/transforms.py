import math
from typing import Tuple
import numpy as np
import cv2
from PIL import Image

def crop_image_depth_and_intrinsic_by_pp(
    image, depth_map, intrinsic, target_shape, track=None, filepath=None, strict=False
):
    """    
    Crops the given image and depth map around the camera's principal point, as defined by `intrinsic`.
    Specifically:
      - Ensures that the crop is centered on (cx, cy).
      - Optionally pads the image (and depth map) if `strict=True` and the result is smaller than `target_shape`.
      - Shifts the camera intrinsic matrix (and `track` if provided) accordingly.

    Args:
        image (np.ndarray):
            Input image array of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array of shape (H, W), or None if not available.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3). The principal point is assumed to be at (intrinsic[1,2], intrinsic[0,2]).
        target_shape (tuple[int, int]):
            Desired output shape: [height, width].
        track (np.ndarray or None):
            Optional array of shape (N, 2). Interpreted as (x, y) pixel coordinates. Will be shifted after cropping.
        filepath (str or None):
            An optional file path for debug logging (only used if strict mode triggers warnings).
        strict (bool):
            If True, will zero-pad to ensure the exact target_shape even if the cropped region is smaller.

    Raises:
        AssertionError:
            If the input image is smaller than `target_shape`.
        ValueError:
            If the cropped image is larger than `target_shape` (in strict mode), which should not normally happen.

    Returns:
        tuple:
            (cropped_image, cropped_depth_map, updated_intrinsic, updated_track)

            - cropped_image (np.ndarray): Cropped (and optionally padded) image.
            - cropped_depth_map (np.ndarray or None): Cropped (and optionally padded) depth map.
            - updated_intrinsic (np.ndarray): Intrinsic matrix adjusted for the crop.
            - updated_track (np.ndarray or None): Track array adjusted for the crop, or None if track was not provided.
    """
    original_h, original_w = image.shape[:2]
    target_h, target_w = target_shape
    intrinsic = np.copy(intrinsic)

    if original_h < target_h:
        error_message = (
            f"Height check failed: original height {original_h} "
            f"is less than target height {target_h}."
        )
        print(error_message)
        raise AssertionError(error_message)

    if original_w < target_w:
        error_message = (
            f"Width check failed: original width {original_w} "
            f"is less than target width {target_w}."
        )
        print(error_message)
        raise AssertionError(error_message)
    
    # # cx 对应 x轴 (Width), cy 对应 y轴 (Height)
    # Identify principal point (cx, cy) from intrinsic
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # Compute how far we can crop in each direction
    if strict:
        half_x = min((target_w / 2), cx)
        half_y = min((target_h / 2), cy)
    else:
        half_x = min((target_w / 2), cx, original_w - cx)
        half_y = min((target_h / 2), cy, original_h - cy)

    # Compute starting indices
    start_x = math.floor(cx) - math.floor(half_x)
    start_y = math.floor(cy) - math.floor(half_y)

    assert start_x >= 0 and start_y >= 0

    # Compute ending indices
    if strict:
        end_x = start_x + target_w
        end_y = start_y + target_h
    else:
        end_x = start_x + 2 * math.floor(half_x)
        end_y = start_y + 2 * math.floor(half_y)

    # Perform the crop
    image = image[start_y:end_y, start_x:end_x, :]
    if depth_map is not None:
        depth_map = depth_map[start_y:end_y, start_x:end_x]

    # Shift the principal point in the intrinsic
    intrinsic[0, 2] = intrinsic[0, 2] - start_x
    intrinsic[1, 2] = intrinsic[1, 2] - start_y

    # Adjust track if provided
    if track is not None:
        track[:, 0] = track[:, 0] - start_x
        track[:, 1] = track[:, 1] - start_y

    # If strict, zero-pad if the new shape is smaller than target_shape
    if strict:
        current_h, current_w = image.shape[:2]
        pad_h = target_h - current_h
        pad_w = target_w - current_w

        if pad_h < 0 or pad_w < 0:
             raise ValueError(f"Cropped larger than target: cur=({current_h},{current_w}), tgt=({target_h},{target_w})")

        if pad_h > 0 or pad_w > 0:
            # np.pad for (axis0/height, axis1/width, axis2/channel)
            image = np.pad(
                image,
                pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant", constant_values=0
            )
            if depth_map is not None:
                depth_map = np.pad(
                    depth_map,
                    pad_width=((0, pad_h), (0, pad_w)),
                    mode="constant", constant_values=0
                )

    return image, depth_map, intrinsic, track


def resize_image_depth_and_intrinsic(
    image,
    depth_map,
    intrinsic,
    target_shape,
    track=None,
    pixel_center=True,
    safe_bound=4,
    enable_random_resize=True,
):
    """
    Resizes the given image and depth map (if provided) to slightly larger than `target_shape`,
    updating the intrinsic matrix (and track array if present). Optionally uses random rescaling
    to create some additional margin (based on `enable_random_resize`).

    Steps:
      1. Compute a scaling factor so that the resized result is at least `target_shape + safe_bound`.
      2. Apply an optional triangular random factor if `enable_random_resize=True`.
      3. Resize the image with LANCZOS if downscaling, BICUBIC if upscaling.
      4. Resize the depth map with nearest-neighbor.
      5. Update the camera intrinsic and track coordinates (if any).

    Args:
        image (np.ndarray):
            Input image array (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array (H, W), or None if unavailable.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3).
        target_shape (np.ndarray or tuple[int, int]):
            Desired final shape (height, width).
        track (np.ndarray or None):
            Optional (N, 2) array of pixel coordinates. Will be scaled.
        pixel_center (bool):
            If True, accounts for 0.5 pixel center shift during resizing.
        safe_bound (int or float):
            Additional margin (in pixels) to add to target_shape before resizing.
        enable_random_resize (bool):
            If True, randomly increase the `safe_bound` within a certain range to simulate augmentation.

    Returns:
        tuple:
            (resized_image, resized_depth_map, updated_intrinsic, updated_track)

            - resized_image (np.ndarray): The resized image.
            - resized_depth_map (np.ndarray or None): The resized depth map.
            - updated_intrinsic (np.ndarray): Camera intrinsic updated for new resolution.
            - updated_track (np.ndarray or None): Track array updated or None if not provided.

    Raises:
        AssertionError:
            If the shapes of the resized image and depth map do not match.
    """
    intrinsic = np.copy(intrinsic)
    original_h, original_w = image.shape[:2]
    original_size = np.array(image.shape[:2], dtype=float)  # H, W

    if enable_random_resize:
        random_boundary = np.random.triangular(0, 0, 0.3)
        safe_bound = safe_bound + random_boundary * target_shape.max()

    resize_scales = (target_shape + safe_bound) / original_size
    max_resize_scale = np.max(resize_scales)
    new_h = int(np.floor(original_h * max_resize_scale))
    new_w = int(np.floor(original_w * max_resize_scale))

    # Convert image to PIL for resizing
    image = Image.fromarray(image)
    resample_method = Image.Resampling.LANCZOS if max_resize_scale < 1 else Image.Resampling.BICUBIC
    image = image.resize((new_w, new_h), resample=resample_method)
    image = np.array(image)

    if depth_map is not None:
        depth_map = cv2.resize(
            depth_map,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )

    # resize intrinsic matrix
    actual_scale_x = new_w / original_w
    actual_scale_y = new_h / original_h

    # Old (0,0) is center of top-left pixel.
    # We want coordinate system to verify: pixel (u,v) corresponds to ray d*(K_inv * [u,v,1])
    # Usually coordinate systems assume (0.5, 0.5) is center of top-left pixel for projection.
    if pixel_center:
        intrinsic[0, 2] += 0.5
        intrinsic[1, 2] += 0.5

    intrinsic[0, 0] *= actual_scale_x # fx
    intrinsic[1, 1] *= actual_scale_y # fy
    intrinsic[0, 2] *= actual_scale_x # cx
    intrinsic[1, 2] *= actual_scale_y # cy

    if track is not None:
        track[..., 0] *= actual_scale_x
        track[..., 1] *= actual_scale_y

    if pixel_center:
        intrinsic[0, 2] -= 0.5
        intrinsic[1, 2] -= 0.5

    return image, depth_map, intrinsic, track
