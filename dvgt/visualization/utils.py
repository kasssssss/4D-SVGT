import numpy as np
import cv2
import os
import requests
import pickle
import open3d as o3d
import viser.transforms as viser_tf
from typing import Tuple, Union
from numbers import Number
import copy
import onnxruntime
import trimesh
import matplotlib.cm as cm
import torch

from tools.data_process.utils.io import write_image, write_depth

def depth_to_color(
    depth: torch.Tensor,
    max_depth: float = 200,
): 

    # 拼接V维度到W维度: (V, H, W) -> (H, W * V)
    depth = depth.permute(1, 0, 2).flatten(-2, -1)

    depth = torch.clamp(depth, min=0, max=max_depth)
    depth = depth / (max_depth + 1e-8)
    if depth.requires_grad:
        depth = depth.detach()
    depth = depth.cpu().numpy()
    colored_depth = cm.viridis(depth)[..., :3] # 取RGB通道
    
    return colored_depth.transpose(2, 0, 1)     # (H, W, 3)

def save_point_cloud(points, colors, file_path, save_type='glb'):
    if save_type == 'glb':
        if len(points.shape) != 2:
            points = points.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
        file_path.mkdir(parents=True, exist_ok=True)
        output_path = file_path / "pred_point.glb"
        pcd = trimesh.PointCloud(vertices=points, colors=colors)
        pcd.export(str(output_path))
    elif save_type == 'o3d_pkl':
        if len(points.shape) != 2:
            points = points.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        
        file_path.mkdir(parents=True, exist_ok=True)
        output_path = file_path / f"pred_point_o3d.pkl"
        o3d.io.write_point_cloud(str(output_path), pcd)
    else:
        raise NotImplementedError()
    print(f"Point cloud saved to: {output_path}")

def save_images(images, visual_base_dir, cam_types):
    T, V, H, W, _ = images.shape
    for t in range(T):
        image_frame_path = visual_base_dir / f'frame_{t:04d}'
        image_frame_path.mkdir(parents=True, exist_ok=True)
        for v in range(V):
            image_path = image_frame_path / f'{cam_types[v]}.jpg'
            write_image(image_path, images[t, v])

def save_depths(depths, visual_base_dir, cam_types):
    T, V, H, W = depths.shape
    for t in range(T):
        depth_frame_path = visual_base_dir / f'frame_{t:04d}'
        depth_frame_path.mkdir(parents=True, exist_ok=True)
        for v in range(V):
            depth_path = depth_frame_path / f'{cam_types[v]}.png'
            write_depth(depth_path, depths[t, v])

def center_data(points, poses):
    """
    Translate the point cloud and poses to a new coordinate system 
    where the point cloud's centroid is the origin.
    """
    if points.shape[0] == 0:
        return points, poses, np.zeros(3)
        
    center = np.mean(points, axis=0)
    points_centered = points - center
    
    poses_centered = poses.copy()
    poses_centered[..., :3, 3] -= center
    
    print(f"Data centered. Original center was at {center}")
    return points_centered, poses_centered, center

def process_and_filter_points(points, colors, mask, max_depth, downsample_ratio):
    if max_depth > 0:
        print(f"Truncating points beyond max depth of {max_depth}m...")
        depth = np.linalg.norm(points, axis=-1)
        depth_mask = depth <= max_depth
        mask = mask & depth_mask

    if mask is not None:
        points = points[mask]
        colors = colors[mask]
    else:
        # if mask is None，still needs flatten
        points = points.reshape(-1, 3)
        colors = colors.reshape(-1, 3)

    if 0 < downsample_ratio < 1.0:
        num_points = points.shape[0]
        target_num_points = int(num_points * downsample_ratio)
        if target_num_points < num_points and target_num_points > 0:
            print(f"Downsampling from {num_points} to {target_num_points} points...")
            indices = np.random.choice(num_points, target_num_points, replace=False)
            points = points[indices]
            colors = colors[indices]

    return points, colors


def load_pickle_data(path):
    print(f"Loading data from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Handle Batch dimension = 1 automatically
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            cleaned_data[key] = value[0]
        elif isinstance(value, list):
            if isinstance(value[0], list):
                cleaned_data[key] = [v[0] for v in value]
            else:
                cleaned_data[key] = value[0]
        else:
            cleaned_data[key] = value

    # 图像转置 (C, H, W) -> (H, W, C)
    images = np.transpose(cleaned_data['images'], (0, 1, 3, 4, 2))
    cleaned_data['images'] = (images * 255).astype(np.uint8) if images.max() <= 1.0 else images.astype(np.uint8)

    return cleaned_data

def segment_sky(image, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image: H, W, 3
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32
    
    if mask_filename is not None:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def apply_sky_segmentation(images: np.ndarray) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (T, V, H, W)
        images (np.ndarray): Image input model with shape (T, V, H, W, 3)

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    T, V, H, W, _ = images.shape

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    sky_mask_list = []

    print("Generating sky masks...")
    for t in range(T):
        for v in range(V):
            sky_mask = segment_sky(images[t, v], skyseg_session)

            # Resize mask to match H×W if needed
            if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                sky_mask = cv2.resize(sky_mask, (W, H))

            sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape TVHW
    sky_mask_array = np.array(sky_mask_list).reshape(T, V, H, W)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1)

    print("Sky segmentation applied successfully")
    return sky_mask_binary

# point and depth edge mask
def angle_diff_vec3_numpy(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12):
    """
    Compute angle difference between 3D vectors using NumPy.
    """
    return np.arctan2(
        np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1) + eps, (v1 * v2).sum(axis=-1)
    )

def points_to_normals(
    point: np.ndarray, mask: np.ndarray = None, edge_threshold: float = None
) -> np.ndarray:
    """
    Calculate normal map from point map. Value range is [-1, 1].
    """
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack(
        [
            np.cross(up, left, axis=-1),
            np.cross(left, down, axis=-1),
            np.cross(down, right, axis=-1),
            np.cross(right, up, axis=-1),
        ]
    )
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    valid = (
        np.stack(
            [
                mask[:-2, 1:-1] & mask[1:-1, :-2],
                mask[1:-1, :-2] & mask[2:, 1:-1],
                mask[2:, 1:-1] & mask[1:-1, 2:],
                mask[1:-1, 2:] & mask[:-2, 1:-1],
            ]
        )
        & mask[None, 1:-1, 1:-1]
    )
    if edge_threshold is not None:
        view_angle = angle_diff_vec3_numpy(pts[None, 1:-1, 1:-1, :], normal)
        view_angle = np.minimum(view_angle, np.pi - view_angle)
        valid = valid & (view_angle < np.deg2rad(edge_threshold))

    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    if has_mask:
        normal_mask = valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    else:
        return normal

def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Create a sliding window view of the input array along a specified axis.
    """
    assert x.shape[axis] >= window_size, (
        f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    )
    axis = axis % x.ndim
    shape = (
        *x.shape[:axis],
        (x.shape[axis] - window_size + 1 + stride -1) // stride,
        *x.shape[axis + 1 :],
        window_size,
    )
    
    axis_size = x.shape[axis]
    n_windows = (axis_size - window_size) // stride + 1
    
    shape = (
        *x.shape[:axis],
        n_windows,
        *x.shape[axis + 1 :],
        window_size,
    )

    strides = (
        *x.strides[:axis],
        stride * x.strides[axis],
        *x.strides[axis + 1 :],
        x.strides[axis],
    )
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding

def sliding_window_nd(
    x: np.ndarray,
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Create sliding windows along multiple dimensions of the input array.
    """
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x

def sliding_window_2d(
    x: np.ndarray,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
) -> np.ndarray:
    """
    Create 2D sliding windows over the input array.
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)

def max_pool_1d(
    x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1
):
    """
    Perform 1D max pooling on the input array.
    """
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        pad_shape = list(x.shape)
        pad_shape[axis] = padding
        padding_arr = np.full(
            tuple(pad_shape),
            fill_value=fill_value,
            dtype=x.dtype,
        )
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool

def max_pool_nd(
    x: np.ndarray,
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Perform N-dimensional max pooling on the input array.
    """
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x

def max_pool_2d(
    x: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
):
    """
    Perform 2D max pooling on the input array.
    """
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)

def depth_edge(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the edge mask from depth map.
    """
    if mask is None:
        diff = max_pool_2d(
            depth, kernel_size, stride=1, padding=kernel_size // 2
        ) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
    else:
        diff = max_pool_2d(
            np.where(mask, depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ) + max_pool_2d(
            np.where(mask, -depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol

    if rtol is not None:
        valid_depth = np.where(depth > 1e-6, depth, 1e-6)
        edge |= diff / valid_depth > rtol
    return edge

def normals_edge(
    normals: np.ndarray, tol: float, kernel_size: int = 3, mask: np.ndarray = None
) -> np.ndarray:
    """
    Compute the edge mask from normal map.
    """
    assert normals.ndim >= 3 and normals.shape[-1] == 3, (
        "normal should be of shape (..., height, width, 3)"
    )
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    padding = kernel_size // 2
    
    pad_width = [(0, 0)] * normals.ndim
    pad_width[normals.ndim - 3] = (padding, padding) # H
    pad_width[normals.ndim - 2] = (padding, padding) # W
    
    normals_padded = np.pad(
        normals,
        pad_width,
        mode="edge",
    )

    normals_window = sliding_window_2d(
        normals_padded,
        window_size=kernel_size,
        stride=1,
        axis=(-3, -2), # H, W axes
    )
    
    # normals shape: (H, W, 3) -> (H, W, 3, 1, 1)
    # normals_window shape: (H, W, 3, K, K)
    normals_expanded = normals[..., None, None]
    
    # (H, W, K, K)
    dot_prod = (normals_expanded * normals_window).sum(axis=-3)
    # clip to prevent arccos domain errors
    dot_prod_clipped = np.clip(dot_prod, -1.0, 1.0)
    angle_diff = np.arccos(dot_prod_clipped)

    if mask is not None:
        mask_pad_width = [(0, 0)] * mask.ndim
        mask_pad_width[mask.ndim - 2] = (padding, padding) # H
        mask_pad_width[mask.ndim - 1] = (padding, padding) # W
        
        mask_padded = np.pad(
            mask,
            mask_pad_width,
            mode="edge",
        )
        
        mask_window = sliding_window_2d(
            mask_padded,
            window_size=kernel_size,
            stride=1,
            axis=(-2, -1), # H, W axes
        )
        # angle_diff (H, W, K, K), mask_window (H, W, K, K)
        angle_diff = np.where(
            mask_window,
            angle_diff,
            0,
        )
        
    angle_diff = angle_diff.max(axis=(-2, -1))

    # The original implementation seems to have an extra max_pool, which might be for dilating the edges.
    # Replicating it here.
    angle_diff = max_pool_2d(
        angle_diff, kernel_size, stride=1, padding=kernel_size // 2
    )
    edge = angle_diff > np.deg2rad(tol)
    return edge

def visualize_ego_poses(server, poses, frustum_scale=0.1, name_prefix="ego_pose"):
    print(f"Visualizing {len(poses)} ego poses...")
    for i, pose in enumerate(poses):
        # pose: ego_n_to_ego_first, 4×4
        T_world_camera = viser_tf.SE3.from_matrix(pose)
        
        server.scene.add_camera_frustum(
            f"/{name_prefix}/{i}",
            fov=np.pi / 4,  # set by default
            aspect=1.77,    # set by default
            scale=frustum_scale,
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            color=(255, 100, 100)
        )
