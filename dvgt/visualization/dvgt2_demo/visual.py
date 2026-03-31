import pickle
import time
import numpy as np
import viser
from scipy.interpolate import CubicSpline

def create_trajectory_ribbon(
    server, 
    name, 
    traj_8x3, 
    width=2, 
    color=(0.4, 0.7, 1.0),
    forward_offset=0,  # 往前平移量 (单位: 米)
    up_offset=-0.1       # 往上平移量，稍微抬高即可避免路面遮挡
):
    """
    使用三次样条插值生成平滑轨迹，并渲染为带加粗边界的浅蓝色平面带(Ribbon)
    轨迹是RF，最后一个维度是theta，直接扔掉
    """
    # traj_8x3[:, :2] = traj_8x3[:, :2] / 0.1  # 转化单位
    traj_rf = np.concatenate([np.array([[0, 0]]), traj_8x3[:, :2]])

    # 1. 样条插值 (从 8 个点插值到 50 个点)
    t_orig = np.linspace(0, 1, len(traj_rf))
    t_new = np.linspace(0, 1, 50)
    cs = CubicSpline(t_orig, traj_rf)
    traj_interp = cs(t_new) # (50, 2)

    # --- 坐标系修正与平移 ---
    traj_interp = np.stack([    # 转为rdf
        traj_interp[:, 0],
        np.zeros_like(traj_interp[:, 0]),
        traj_interp[:, 1],
    ], axis=-1)
    traj_interp[:, 2] += forward_offset
    traj_interp[:, 1] += up_offset

    # 2 & 3. 抛弃容易受噪声干扰的法线计算，直接沿 Right 轴 (X轴) 左右平移
    left_bound = np.copy(traj_interp)
    left_bound[:, 0] -= width / 2   # 向左平移 (X轴减小)

    right_bound = np.copy(traj_interp)
    right_bound[:, 0] += width / 2  # 向右平移 (X轴增加)

    # 4. 构建 Mesh 面片 (用于中间浅蓝色平面)
    vertices = np.concatenate([left_bound, right_bound], axis=0)
    n = len(traj_interp)
    faces = []
    for i in range(n - 1):
        # 构成两个三角形拼接成矩形
        faces.append([i, i + n, i + 1])
        faces.append([i + n, i + n + 1, i + 1])
    faces = np.array(faces)

    # 渲染中间平面 (浅蓝色透明)
    server.scene.add_mesh_simple(
        name=f"{name}/plane",
        position=(0, 0, 0),
        vertices=vertices,
        faces=faces,
        color=color,
        opacity=0.5,
        side="double"
    )
    
    # 渲染加粗的左右边界线，使用 add_spline_catmull_rom
    border_color = (0.1, 0.3, 0.8) # 深蓝色边界
    server.scene.add_spline_catmull_rom(
        name=f"{name}/border_left",
        positions=left_bound,
        color=border_color,
        line_width=4.0
    )
    server.scene.add_spline_catmull_rom(
        name=f"{name}/border_right",
        positions=right_bound,
        color=border_color,
        line_width=4.0
    )

def read_streamed_pkl(file_path):
    processed_frames = []
    
    with open(file_path, 'rb') as f:
        while True:
            try:
                # 每次 load 会自动读取下一个完整的数据块
                frame_data = pickle.load(f)
                processed_frames.append(frame_data)
            except EOFError:
                # 读到文件末尾，退出循环
                break
                
    return processed_frames


def main():
    data_path = 'path/to/you/pkl'
    frames_data = read_streamed_pkl(data_path)
    # with open(data_path, 'rb') as f:
    #     frames_data = pickle.load(f)

    car_path = 'dvgt/visualization/visual_dvgt_2/xiaomi_yu7.pkl'
    with open(car_path, 'rb') as f:
        car_data = pickle.load(f)

    server = viser.ViserServer()
    server.configure_theme(show_logo=False, control_layout="collapsible", dark_mode=False)
    
    # GUI 控制面板
    gui_frame = server.gui.add_slider("Frame", min=0, max=len(frames_data)-1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("Play", initial_value=False)
    
    # 初始化相机视角 (固定视角：向后平移并俯视)
    # 因为点云已经是 ego 坐标系，(0,0,0) 就是自车中心
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (-15.0, 0.0, 20.0) # 在车后方 15m，高 20m
        client.camera.look_at = (5.0, 0.0, 0.0)     # 看向车前方 5m 处

    def update_scene(frame_idx):
        data = frames_data[frame_idx]
        
        # 1. 更新当前窗口的点云
        server.scene.add_point_cloud(
            name="scene/point_cloud",
            points=np.concatenate([data['points'], car_data['points']]),
            colors=np.concatenate([data['colors'], car_data['colors']]),
            point_size=0.06,
        )
        
        # 2. 更新轨迹和边界
        create_trajectory_ribbon(server, "scene/trajectory", data['traj'])

    # 监听滑块变化
    @gui_frame.on_update
    def _(_):
        update_scene(gui_frame.value)

    # 初始渲染第一帧
    update_scene(0)

    # 播放循环
    fps = 1
    while True:
        if gui_play.value:
            next_frame = (gui_frame.value + 1) % len(frames_data)
            gui_frame.value = next_frame # 赋值会触发 @gui_frame.on_update
            time.sleep(1.0 / fps)
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    main()