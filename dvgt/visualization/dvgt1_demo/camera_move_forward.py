import numpy as np
import viser
import time
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline

def generate_pose_trajectory(poses, n_steps=200, height=2.0, forward=0, pitch_down_deg=0.0):
    """
    生成平滑插值后的相机轨迹。
    
    Args:
        poses: [N, 3, 4] or [N, 4, 4] 原始位姿矩阵
        n_steps: int, 插值后的总帧数（例如 30fps * 10s = 300）
        height: float, 相机抬高的高度
        forward: float, 相机向前移动的距离
        pitch_down_deg: float, 相机向下俯视的角度
    """
    # 1. 提取原始关键帧数据
    # 位置
    key_positions = poses[:, :3, 3].copy()
    key_positions[:, 1] -= height   # 应用高度调整
    key_positions[:, 2] += forward
    
    # 旋转 (先计算出调整后的最终旋转，再进行插值)
    r_origin = R.from_matrix(poses[:, :3, :3])
    r_pitch = R.from_euler('x', -pitch_down_deg, degrees=True)
    key_rotations = r_origin * r_pitch # 复合旋转
    
    # 2. 准备时间轴
    n_keyframes = len(poses)
    times_orig = np.arange(n_keyframes)             # 原始时间点: 0, 1, 2 ... N-1
    times_new = np.linspace(0, n_keyframes - 1, n_steps) # 新的时间点
    
    # 3. 位置插值 (Cubic Spline - 产生平滑曲线)
    # axis=0 表示沿着 N 的维度进行插值
    cs = CubicSpline(times_orig, key_positions, axis=0) 
    interp_positions = cs(times_new)
    
    # 4. 旋转插值 (Slerp - 产生平滑转动)
    slerp = Slerp(times_orig, key_rotations)
    interp_rotations = slerp(times_new)
    
    # 5. 转换格式输出
    quats_xyzw = interp_rotations.as_quat()
    
    # 调整为 viser 需要的 [w, x, y, z] 顺序
    traj_wxyz = np.stack([
        quats_xyzw[:, 3], # w
        quats_xyzw[:, 0], # x
        quats_xyzw[:, 1], # y
        quats_xyzw[:, 2]  # z
    ], axis=1)
    
    return interp_positions, traj_wxyz

def load_custom_trajectory():
    # traj_positions = my_trajectory_numpy # (N, 3)
    # traj_look_ats = my_look_at_targets   # (N, 3)
    # return traj_positions, traj_look_ats
    pass

# python visual/camera_move_forward.py

if __name__ == "__main__":

    data_path = 'path/to/you/saved_data.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    args_conf_percentile = 25

    images = data['images'][0].transpose(0, 1, 3, 4, 2) 
    pred_points = data['pred_points'][0]      # (T, V, H, W, 3)
    point_masks = data['point_masks'][0]      # (T, V, H, W)
    pred_points_conf = data['pred_points_conf'][0] # (T, V, H, W)

    cutoff = np.percentile(pred_points_conf, args_conf_percentile)
    current_mask = (pred_points_conf >= cutoff) & point_masks 

    pc_colors = images[current_mask]
    pc_points = pred_points[current_mask]

    pose = data['pred_ego_n_to_ego_first'][0]
    print(f"Loaded data shapes - Points: {pc_points.shape}, Colors: {pc_colors.shape}")

    server = viser.ViserServer()

    server.configure_theme(
        # 1. 隐藏左下角的 Viser 图标
        show_logo=False,
        
        # 2. 隐藏右上角的“Share”按钮（通常位于控制台顶部）
        show_share_button=False,
        
        # 3. 隐藏左上角的标题（如果不需要）
        titlebar_content=None,
        
        # 4. 设置控制面板布局
        # "collapsible": 允许折叠（你会看到一个小箭头，折叠后几乎不可见）
        # "floating": 悬浮模式
        # "fixed": 固定模式
        control_layout="collapsible", 
        
        # 可选：如果你想调整背景色以方便抠图或展示
        # dark_mode=True 
    )

    server.scene.add_point_cloud(
        name="move point cloud",
        points=pc_points,
        colors=pc_colors,
        point_size=0.01, # 稍微调小一点，看起来更精致
    )

    # 准备相机轨迹
    traj_positions, traj_wxyz = generate_pose_trajectory(pose)


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
            curr_wxyz = traj_wxyz[step_index]
            
            # 遍历所有连接的客户端并更新相机
            for client_id, client in clients.items():
                client.camera.position = curr_pos
                client.camera.wxyz = curr_wxyz
            
            # 更新索引实现循环
            step_index = (step_index + 1) % total_steps
            
        time.sleep(delay)