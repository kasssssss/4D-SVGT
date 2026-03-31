import torch

def project_lidar_to_depth(points, points_coordinate_to_camera, camera_intrinsic, width, height):
    """
    Fast implementation to project the 3D points to image plane to get the depth.

    points: [N, 3]
    points_coordinate_to_camera: lidar to cam, [4, 4]
    camera_intrinsic: [3, 3]
    max_depth: lidar的有效深度

    return, 没有投影到depth的值为INF
    """
    points_h = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
    points_cam = points_h @ points_coordinate_to_camera.T
    points_cam = points_cam[:, :3]

    valid_mask = (points_cam[:, 2] > 1) & (points_cam[:, 2] < 150)    # 不在这个范围的lidar都认为不可靠
    points_cam = points_cam[valid_mask]

    points_depth = points_cam[:, 2]

    points_uvw = points_cam @ camera_intrinsic.T
    points_uv = points_uvw[:, :2] / points_uvw[:, 2:3]

    u_round = torch.round(points_uv[:, 0]).long()
    v_round = torch.round(points_uv[:, 1]).long()

    valid_uv_mask = (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)
    u_valid = u_round[valid_uv_mask]
    v_valid = v_round[valid_uv_mask]
    z_valid = points_depth[valid_uv_mask]

    indices = v_valid * width + u_valid
    depth_image = torch.full((height * width,), float('inf'), device=points.device, dtype=points.dtype)

    depth_image.scatter_reduce_(0, indices, z_valid.to(depth_image.dtype), "amin", include_self=False)

    depth_image = depth_image.view(height, width)

    return depth_image
