import numpy as np
from typing import List
from pyquaternion import Quaternion

def project_lidar_to_depth(points: np.ndarray, points_coordinate_to_camera, camera_intrinsic, width, height):
    """
    Fast implementation to project the 3D points to image plane to get the depth.

    points: [N, 3]
    points_coordinate_to_camera: lidar to cam, [4, 4]
    camera_intrinsic: [3, 3]
    max_depth: lidar的有效深度

    return, 没有投影到depth的值为INF
    """
    points = points.astype(np.float64)
    points_coordinate_to_camera = points_coordinate_to_camera.astype(np.float64)
    camera_intrinsic = camera_intrinsic.astype(np.float64)
    
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = points_h @ points_coordinate_to_camera.T
    points_cam = points_cam[:, :3]

    valid_mask = (points_cam[:, 2] > 1) & (points_cam[:, 2] < 150)    # 不在这个范围的lidar都认为不可靠
    points_cam = points_cam[valid_mask]

    points_depth = points_cam[:, 2]

    points_uvw = points_cam @ camera_intrinsic.T
    points_uv = points_uvw[:, :2] / points_uvw[:, 2:3]

    u_round = np.round(points_uv[:, 0]).astype(np.int64)
    v_round = np.round(points_uv[:, 1]).astype(np.int64)

    valid_uv_mask = (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)
    u_valid = u_round[valid_uv_mask]
    v_valid = v_round[valid_uv_mask]
    z_valid = points_depth[valid_uv_mask]

    indices = v_valid * width + u_valid
    depth_image = np.full((height * width,), np.inf)

    np.minimum.at(depth_image, indices, z_valid)

    depth_image = depth_image.reshape(height, width)

    return depth_image

def build_transform_matrix(translation: List[float], rotation: List[float]) -> np.ndarray:
    """ 从 translation 和 quaternion rotation 构建 4x4 变换矩阵 """
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = np.asarray(translation)
    return T