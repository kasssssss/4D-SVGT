import pickle
import time
import numpy as np
import viser
from einops import rearrange

def compute_ipm_points(image, K, ego2cam, step=4, max_dist=40.0, ground_y=1.8):
    """
    基于平地假设，在 RDF (Right-Down-Forward) 坐标系下将 2D 点投影到 3D 地面
    """
    H, W, _ = image.shape
    
    # 1. 构造图像网格 (u, v)
    u, v = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3) # (N, 3)

    # 2. 转换到相机坐标系下的射线方向 d_cam
    K_inv = np.linalg.inv(K)
    d_cam = (K_inv @ uv1.T).T # (N, 3)

    # 3. 转换到 Ego 坐标系下的射线
    cam2ego = np.linalg.inv(ego2cam)
    R_c2e = cam2ego[:3, :3]
    t_c2e = cam2ego[:3, 3] # [X_right, Y_down, Z_forward]

    d_ego = (R_c2e @ d_cam.T).T # 射线方向 (N, 3)
    O_ego = t_c2e               # 射线起点 (3, )

    # 4. RDF 坐标系下求射线与地面 Y = ground_y 的交点
    # 地面的法向量是 Y 轴，所以我们看 d_ego 的第 1 个维度 (Y)
    d_y = d_ego[:, 1]
    
    # 过滤掉朝天上看、或者平行的射线。因为 Y 向下为正，射线必须 d_y > 0 才能打到地面
    valid_mask = d_y > 1e-5 

    # 计算缩放系数 lambda: O_y + lambda * d_y = ground_y 
    # 使用 np.where 避免除以 0 导致 warning，反正无效的点会被 mask 掉
    safe_d_y = np.where(valid_mask, d_y, 1.0)
    lambdas = (ground_y - O_ego[1]) / safe_d_y
    
    # 过滤掉摄像机背后的点，以及距离 Ego 太远的噪点
    valid_mask = valid_mask & (lambdas > 0) & (lambdas < max_dist)

    lambdas = lambdas[valid_mask]
    d_ego_valid = d_ego[valid_mask]
    
    # 计算出最终的 3D 地面点
    points_3d = O_ego + lambdas[:, None] * d_ego_valid
    
    # 获取对应的颜色值，并将 0~1 转换为 0~255 的 uint8 格式
    colors = image[v.flatten()[valid_mask], u.flatten()[valid_mask], :]
    colors = (colors * 255).astype(np.uint8)
    
    return points_3d, colors

def main():
    # 1. 读取数据
    data_path = 'output/dvgt2/visual/geometry/default/openscene/fix_40_future_8/openscene#log-0104-scene-0007#0#cliplen#40/saved_data.pkl'
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    images = rearrange(data['images'][0], 't v c h w -> t v h w c')                
    intrinsics = data['intrinsics'][0]         
    extrinsics = data['ego_first_to_cam_n'][0] 
    
    num_frames = images.shape[0]
    num_cams = images.shape[1]
    print(f"Loaded {num_frames} frames, {num_cams} cameras per frame.")

    # 2. 初始化 Viser
    server = viser.ViserServer()
    server.configure_theme(show_logo=False, control_layout="collapsible")
    
    # GUI 控制
    gui_frame = server.gui.add_slider("Frame", min=0, max=num_frames-1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("Play", initial_value=False)
    gui_step = server.gui.add_slider("Downsample Step", min=1, max=10, step=1, initial_value=4)
    gui_max_dist = server.gui.add_slider("Max Distance", min=10.0, max=100.0, step=5.0, initial_value=40.0)
    
    # 新增：调整地面高度的滑块 (默认车顶到地面通常在 1.5m ~ 2.0m 之间)
    gui_ground_y = server.gui.add_slider("Ground Y (Height)", min=0.0, max=3.0, step=0.05, initial_value=1.8)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Viser 默认是 Z 向上。而我们的点云是 RDF (Y向下, Z向前)。
        # 因此，要看清这个点云，相机的 Y 需要是一个负数（飞到天上去俯视），看往原点
        client.camera.position = (0.0, -15.0, -5.0) 
        client.camera.look_at = (0.0, 1.8, 10.0)    

    def update_scene():
        frame_idx = gui_frame.value
        step = gui_step.value
        max_dist = gui_max_dist.value
        ground_y = gui_ground_y.value
        
        all_points = []
        all_colors = []
        
        for v in range(num_cams):
            img = images[frame_idx, v]
            K = intrinsics[frame_idx, v]
            ego2cam = extrinsics[frame_idx, v]
            
            pts, cols = compute_ipm_points(img, K, ego2cam, step=step, max_dist=max_dist, ground_y=ground_y)
            all_points.append(pts)
            all_colors.append(cols)
            
        if len(all_points) > 0:
            all_points = np.concatenate(all_points, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)
            
            server.scene.add_point_cloud(
                name="scene/ipm_cloud",
                points=all_points,
                colors=all_colors,
                point_size=0.06,
            )

    @gui_frame.on_update
    def _(_): update_scene()
    
    @gui_step.on_update
    def _(_): update_scene()
    
    @gui_max_dist.on_update
    def _(_): update_scene()
    
    @gui_ground_y.on_update
    def _(_): update_scene()

    update_scene()

    fps = 5 
    while True:
        if gui_play.value:
            next_frame = (gui_frame.value + 1) % num_frames
            gui_frame.value = next_frame 
            time.sleep(1.0 / fps)
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    main()