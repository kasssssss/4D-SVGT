import argparse
from einops import rearrange
import numpy as np
import viser
import viser.transforms as viser_tf
from time import sleep
import torch
import numpy as np

from dvgt.models.architectures.dvgt1 import DVGT1
# from dvgt.models.architectures.dvgt2 import DVGT2
from dvgt.utils.load_fn import load_and_preprocess_images
from iopath.common.file_io import g_pathmgr
from dvgt.evaluation.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n
from dvgt.utils.pose_encoding import decode_pose
from dvgt.visualization.utils import apply_sky_segmentation, depth_edge, points_to_normals, \
    normals_edge, visualize_ego_poses, process_and_filter_points, center_data

def visualize_pred(predictions, args):
    B, T, V, H, W, _ = predictions['points'].shape
    assert B == 1, "Only batch size = 1 is supported for the visualization."

    pred_ego_n_to_ego_0, _ = decode_pose(predictions['absolute_ego_pose_enc'])

    pred_points = predictions['points']
    pred_ray_depth_in_ego_n = convert_point_in_ego_0_to_ray_depth_in_ego_n(pred_points, pred_ego_n_to_ego_0)

    # Squeeze the batch dimension for visualization.
    pred_points = pred_points[0].cpu().numpy()
    pred_points_conf = predictions['points_conf'][0].cpu().numpy()
    pred_depth = pred_ray_depth_in_ego_n[0].cpu().numpy()       # for depth edge mask
    pred_ego_poses = pred_ego_n_to_ego_0[0].cpu().numpy()
    images = rearrange(predictions['images'][0].cpu().numpy(), 't v c h w -> t v h w c') * 255   # T, V, H, W, 3
    images = images.astype(np.uint8)

    # construct the combined mask
    combined_mask = np.ones((T, V, H, W), dtype=bool)   

    if args.conf_threshold > 0:
        cutoff_value = np.percentile(pred_points_conf, args.conf_threshold)
        conf_mask = pred_points_conf >= cutoff_value
        combined_mask &= conf_mask

    if args.mask_sky:
        sky_mask = apply_sky_segmentation(pred_points_conf, images) 
        combined_mask &= sky_mask
    
    if args.use_edge_masks:
        # Applying edge masks
        edge_mask = np.ones_like(combined_mask)

        for t_idx in range(T):
            for v_idx in range(V):
                frame_pts = pred_points[t_idx, v_idx]  # (H, W, 3)
                frame_depth = pred_depth[t_idx, v_idx] # (H, W)
                frame_base_mask = combined_mask[t_idx, v_idx] # (H, W)

                # 1. Depth Edge Mask
                depth_edges = depth_edge(
                    frame_depth, 
                    rtol=args.edge_depth_rtol, 
                    mask=frame_base_mask
                )
                
                # 2. Normal Edge Mask
                normals, normals_mask = points_to_normals(
                    frame_pts, mask=frame_base_mask
                )
                normal_edges = normals_edge(
                    normals, tol=args.edge_normal_tol, mask=normals_mask
                )

                edge_mask[t_idx, v_idx] = ~(depth_edges | normal_edges)
            
            combined_mask &= edge_mask

    points_final, colors_final = process_and_filter_points(
        pred_points, images, combined_mask, args.max_depth, args.downsample_ratio
    )
    
    points_centered, poses_centered, center = center_data(points_final, pred_ego_poses)

    return [(points_centered, colors_final, "pred_point_cloud")], poses_centered

def main():
    parser = argparse.ArgumentParser(description="Autonomous Driving Scene Point Cloud Visualizer")
    parser.add_argument(
        "--model_name", type=str, default="DVGT1", choices=['DVGT1', 'DVGT2'])
    parser.add_argument(
        "--image_folder", type=str, default="visual_demo_examples/openscene_log-0104-scene-0007", help="Path to folder containing images"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="ckpt/dvgt1.pt", help="Path to folder containing images"
    )
    parser.add_argument("--start_frame", type=int, default=0, help="The start frame in the example autonomous video.")
    parser.add_argument("--end_frame", type=int, default=4, help="The end frame in the example autonomous video.")

    parser.add_argument('--downsample_ratio', type=float, default=-1, help="Random downsample ratio (0.0 to 1.0). Default: -1 (no downsampling).")
    parser.add_argument('--max_depth', type=float, default=-1, help="Maximum depth of points to visualize in meters. Default: -1 (no truncation).")

    parser.add_argument('--no_ego', action='store_true', help="Disable ego pose visualization.")
    parser.add_argument('--use_edge_masks', action='store_true', help="Enable depth and normal edge masks.")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")

    parser.add_argument('--edge_depth_rtol', type=float, default=0.1, help="Relative tolerance (rtol) for depth edge detection.")
    parser.add_argument('--edge_normal_tol', type=float, default=50, help="Angle tolerance (degrees) for normal edge detection.")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")

    args = parser.parse_args()

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    model = DVGT1()
    with g_pathmgr.open(args.checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    # Load and preprocess example images (replace with your own image paths)
    images = load_and_preprocess_images(args.image_folder, start_frame=args.start_frame, end_frame=args.end_frame).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
    
    point_clouds_to_show, poses = visualize_pred(predictions, args)

    server = viser.ViserServer()
    print("\n--- Starting Viser Server ---")

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print(f"Client {client.client_id} connected.")
        
        with client.gui.add_folder("Camera Info"):
            gui_cam_wxyz = client.gui.add_text("Cam WXYZ (quat)", initial_value="...")
            gui_cam_pos = client.gui.add_text("Cam Position (xyz)", initial_value="...")

        @client.camera.on_update
        def _(camera: viser.CameraHandle):
            """Callback for camera updates."""
            gui_cam_wxyz.value = str(np.round(camera.wxyz, 3))
            gui_cam_pos.value = str(np.round(camera.position, 3))
        
        _(client.camera)

    with server.gui.add_folder("Controls"):
        point_size_slider = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.1,
            step=0.01,
            initial_value=0.01,
        )

    pc_handles = []
    for points, colors, name in point_clouds_to_show:
        if points.shape[0] == 0: continue
        print(f"point shape: {points.shape}")
        print(f"point mean: {np.mean(points, axis=0)}")
        handle = server.scene.add_point_cloud(
            name=f"/{name}",
            points=points,
            colors=colors,
            point_size=point_size_slider.value,
        )
        pc_handles.append(handle)

    @point_size_slider.on_update
    def _(_) -> None:
        """Update point size for all point clouds."""
        for handle in pc_handles:
            handle.point_size = point_size_slider.value

    if not args.no_ego:
        visualize_ego_poses(server, poses)

    print("\nViser server running. Open the link in your browser.")
    while True:
        sleep(1)

if __name__ == "__main__":
    main()