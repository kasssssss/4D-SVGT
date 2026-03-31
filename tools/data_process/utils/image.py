import cv2
import numpy as np
from typing import Optional, Tuple
from pandas.core.computation.ops import Op
import torch
import math
import logging
logging = logging.getLogger(__name__)

def principal_point_patch_crop(image: np.ndarray, K: np.ndarray, patch_size: int = 14) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, slice]]:
    """
    以主点为中心裁剪图像，并且保证边长是14/16的倍数
    image: [H, W, C]
    K: [3, 3]

    return：indice 裁剪图的索引，后续的depth也可以使用这个裁剪
    """
    K = K.copy()
    original_h, original_w = image.shape[:2]
    cx, cy = K[0, 2], K[1, 2]

    # 计算以主点为中心的最大对称裁剪区域的半宽和半高
    max_half_w = int(min(cx, original_w - cx))
    max_half_h = int(min(cy, original_h - cy))
    
    # 向下取整到14的倍数，这里patch_size是2的倍数
    final_w = math.floor(max_half_w * 2 / patch_size) * patch_size
    final_h = math.floor(max_half_h * 2 / patch_size) * patch_size
    
    # 这里patch size是2的倍数，所以一定可以整除
    half_w = final_w // 2
    half_h = final_h // 2
    
    # 计算裁剪框
    cx_int, cy_int = int(round(cx)), int(round(cy))
    start_y = cy_int - half_h
    end_y = cy_int + half_h
    start_x = cx_int - half_w
    end_x = cx_int + half_w

    indice = (slice(start_y, end_y), slice(start_x, end_x))
    cropped_image = image[indice]

    updated_K = K.copy()
    updated_K[0, 2] = cx - start_x
    updated_K[1, 2] = cy - start_y

    return cropped_image, updated_K, indice

def undistort_image(image: np.ndarray, dist_coeffs: np.ndarray, K: np.ndarray):
    """
    去畸变并裁剪黑边
    image: [H, W, C]
    K: [3, 3]
    """
    h, w = image.shape[:2]
    optimal_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0)
    undistorted_image = cv2.undistort(image, K, dist_coeffs, None, optimal_K)
    # 裁剪图像
    x, y, w_roi, h_roi = roi
    undistorted_image = undistorted_image[y:y+h_roi, x:x+w_roi]
    optimal_K[0, 2] -= x
    optimal_K[1, 2] -= y
    return undistorted_image, optimal_K

def equalize_focal_lengths(image: np.ndarray, K: np.ndarray):
    # 使图片的fx = fy
    fx, fy = K[0, 0], K[1, 1]
    h, w = image.shape[:2]
    new_K = K.copy()
    if not np.isclose(fx, fy):
        f_new = (fx + fy) / 2.0
        scale_x = f_new / fx
        scale_y = f_new / fy
        new_w = int(w * scale_x)
        new_h = int(h * scale_y)
        # 根据 (fx + fy)/2 > \sqrt{fx * fy}，像素的总数总是增加的，所以选择：INTER_CUBIC
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        new_K[0, 0] = f_new
        new_K[1, 1] = f_new
        new_K[0, 2] *= scale_x
        new_K[1, 2] *= scale_y
    return image, new_K

def process_image_sequentially(
    image: np.ndarray, 
    K: np.ndarray, 
    dist_coeffs: Optional[np.ndarray] =None, 
    num_tokens: int = 2100, 
    patch_size: int = 14,
    is_resize_fx_fy: bool = True,
):
    """
    该函数会执行以下四个操作：
    1. 对图像进行去畸变处理，并裁剪掉因去畸变产生的黑边。
    2. resize使得fx = fy。(可选)
    3. 根据MoGe的num token要求，等比resize
    4. crop，使得主点在图片中心 and 满足patch=14的输入

    image [H, W, C]
    K [3, 3]
    dist_coeffs: None or [5]
    """
    new_K = K.copy()
    new_image = image.copy()
    
    if dist_coeffs is not None:
        new_image, new_K = undistort_image(new_image, dist_coeffs, new_K)

    if is_resize_fx_fy:
        new_image, new_K = equalize_focal_lengths(new_image, new_K)
    
    resized_image, resized_K = resize_image_by_num_tokens(new_image, new_K, num_tokens)

    cropped_image, cropped_K, depth_indice = principal_point_patch_crop(resized_image, resized_K, patch_size)
    
    while cropped_image.shape[0] / 14 * cropped_image.shape[1] / 14 < 2000:     # 我们需要预留一些空间，后续还会resize下采样，使得fx != fy
        num_tokens += 100
        logging.debug(f"Cropped image num_token is {int(cropped_image.shape[0] / 14 * cropped_image.shape[1] / 14)} too small, use larger num tokens: {num_tokens}")
        resized_image, resized_K = resize_image_by_num_tokens(cropped_image, cropped_K, num_tokens)
        cropped_image, cropped_K, depth_indice = principal_point_patch_crop(resized_image, resized_K, patch_size)

    return cropped_image, cropped_K, depth_indice

def resize_image_by_num_tokens(image: np.ndarray, K: np.ndarray, num_tokens: int):
    """ 
    严格保证图片宽高比，将图片处理到比num_token要求的大小稍大，同时缩放内参
    """
    new_K = K.copy()
    orig_h, orig_w = image.shape[:2]
    aspect_ratio = orig_w / orig_h
    new_h = math.ceil((num_tokens / aspect_ratio) ** 0.5) * 14  # 确保比num_token要求的稍大
    new_w = math.ceil(new_h * aspect_ratio)
    if new_h != orig_h:
        scale = new_h / orig_h
        new_K[0, 0] *= scale
        new_K[1, 1] *= scale
        new_K[0, 2] *= scale
        new_K[1, 2] *= scale
        # Resize the image with LANCZOS if downscaling, BICUBIC if upscaling.
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4 if new_h < orig_h else cv2.INTER_CUBIC)
    return image, new_K

def get_fov_x_deg(width: int, fx: float) -> float:
    fov_x_rad = 2 * np.arctan2(width / 2, fx)
    fov_x_deg = np.rad2deg(fov_x_rad)
    return fov_x_deg

def sample_sparse_depth(proj_depth: torch.Tensor, proj_depth_mask: Optional[torch.Tensor] = None, sample_count=25000) -> torch.Tensor:
    # gt depth 点太多了，随机采样部分
    if proj_depth_mask is None:
        proj_depth_mask = torch.isfinite(proj_depth)

    finite_indices = torch.nonzero(proj_depth_mask)
    num_finite = finite_indices.shape[0]
    if num_finite > sample_count:
        choice_indices = torch.randperm(num_finite)[:sample_count]
        sampled_indices = finite_indices[choice_indices]
        rows, cols = sampled_indices.T
        result_depth = torch.full_like(proj_depth, float('inf'))
        result_depth[rows, cols] = proj_depth[rows, cols]
        proj_depth = result_depth
    return proj_depth

def resize_to_aspect_ratio(image: np.ndarray, K: np.ndarray, original_aspect_ratio: float):
    """
    将图像和内参resize回原本的宽高比，fx != fy
    """
    image_save = image.copy()
    K_save = K.copy()
    h, w = image.shape[:2]
    current_aspect_ratio = w / h
    if np.isclose(current_aspect_ratio, original_aspect_ratio):
        return image_save, K_save

    # 这里我们确保resize后的图片像素总数小于当前图片
    # 这样后续对depth可以只进行下采样
    if current_aspect_ratio > original_aspect_ratio:
        new_h = h
        new_w = int(round(new_h * original_aspect_ratio))
    else:
        new_w = w
        new_h = int(round(new_w / original_aspect_ratio))

    image_save = cv2.resize(image_save, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    scale_w = new_w / w
    scale_h = new_h / h
    K_save[0, 0] *= scale_w
    K_save[0, 2] *= scale_w
    K_save[1, 1] *= scale_h
    K_save[1, 2] *= scale_h
        
    return image_save, K_save

def crop_black_borders_with_data(img, K, depth, crop_margin=50):
    """
    固定裁剪图像左右两侧的黑边，并更新内参和深度图。
    
    参数:
        img: 输入图像 (H, W, 3)
        K: 相机内参矩阵 (3, 3)
        depth: 深度图 (H, W)
        crop_margin: 左右两边各要切掉的像素数 (int)，例如 50
        
    返回:
        img_crop: 裁剪后的图像
        K_new: 更新后的内参
        depth_crop: 裁剪后的深度图
    """
    h, w = img.shape[:2]
    
    # 1. 计算裁剪范围
    # 从左边 crop_margin 开始，到右边 w - crop_margin 结束
    x_start = int(crop_margin)
    x_end = int(w - crop_margin)
    
    # 安全检查：防止切没了
    if x_start >= x_end:
        raise ValueError(f"裁剪数值太大！图片宽 {w}，你试图左右各切 {crop_margin}，这就切完了。")

    # 2. 执行切片操作 (Slicing)
    # [高:不变, 宽:裁剪]
    img_crop = img[:, x_start:x_end]
    depth_crop = depth[:, x_start:x_end]
    
    # 3. 更新内参 K
    # 只有 cx (主点 x 坐标) 需要平移，fy, fy, cy 都不变
    K_new = K.copy()
    K_new[0, 2] = K[0, 2] - x_start  # cx_new = cx_old - 左边切掉的量

    return img_crop, K_new, depth_crop