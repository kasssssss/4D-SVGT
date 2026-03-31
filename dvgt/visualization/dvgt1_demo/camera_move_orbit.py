import numpy as np
import viser
import time
import pickle

def generate_orbit_trajectory(radius=5.0, height=10.0, num_steps=500):
    """
    生成一个围绕原点的圆形轨迹。
    height: 圆形轨迹的高度。
    返回:
        positions: (N, 3) 相机位置
        look_ats: (N, 3) 相机看向的目标点 (这里默认一直看原点)
    """
    thetas = np.linspace(0, 2 * np.pi, num_steps)
    # 我们坐标系是RDF，x向右，y向下，z向前
    # xz是水平面
    # y的高度需要是负的
    x = radius * np.cos(thetas)
    z = radius * np.sin(thetas)
    y = np.full_like(x, - height)
    
    positions = np.stack([x, y, z], axis=1) # (N, 3)
    look_ats = np.zeros_like(positions)     # (N, 3) 一直看向 (0,0,0)
    
    return positions, look_ats

def load_custom_trajectory():
    # traj_positions = my_trajectory_numpy # (N, 3)
    # traj_look_ats = my_look_at_targets   # (N, 3)
    # return traj_positions, traj_look_ats
    pass

# python visual/camera_move_orbit.py

if __name__ == "__main__":
    try:
        data_path = 'path/to/you/saved_data.pkl'
        args_conf_percentile = 25
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        images = data['images'][0].transpose(0, 1, 3, 4, 2) 
        pred_points = data['pred_points'][0]      # (T, V, H, W, 3)
        point_masks = data['point_masks'][0]      # (T, V, H, W)
        pred_points_conf = data['pred_points_conf'][0] # (T, V, H, W)
        cutoff = np.percentile(pred_points_conf, args_conf_percentile)
        current_mask = (pred_points_conf >= cutoff) & point_masks 
        pc_colors = images[current_mask]
        pc_points = pred_points[current_mask]
        print(f"Loaded data shapes - Points: {pc_points.shape}, Colors: {pc_colors.shape}")
    except FileNotFoundError:
        print("⚠️ 未找到文件，生成随机数据用于演示...")
        pc_points = np.random.randn(10000, 3)
        pc_colors = np.random.rand(10000, 3)

    server = viser.ViserServer()
    
    server.scene.add_point_cloud(
        name="move point cloud",
        points=pc_points,
        colors=pc_colors,
        point_size=0.001, # 稍微调小一点，看起来更精致
    )

    # 准备相机轨迹
    if True:
        # 这里根据你的数据范围动态计算一个半径，或者你直接传入你的 N,3 轨迹
        data_radius = np.max(np.abs(pc_points)) / 8
        traj_positions, traj_look_ats = generate_orbit_trajectory(radius=data_radius, num_steps=600)
    else:
        traj_positions, traj_look_ats = load_custom_trajectory()

    # 循环播放轨迹
    step_index = 0
    total_steps = len(traj_positions)
    fps = 30
    delay = 1.0 / fps

    while True:
        # 获取当前所有的客户端（浏览器窗口）
        clients = server.get_clients()
        
        if len(clients) > 0:
            # 获取当前帧的相机位姿
            curr_pos = traj_positions[step_index]
            curr_look_at = traj_look_ats[step_index]
            
            # 遍历所有连接的客户端并更新相机
            for client_id, client in clients.items():
                client.camera.position = curr_pos
                client.camera.look_at = curr_look_at
                client.camera.up_direction = (0.0, -1.0, 0.0)
            
            # 更新索引实现循环
            step_index = (step_index + 1) % total_steps
            
        time.sleep(delay)