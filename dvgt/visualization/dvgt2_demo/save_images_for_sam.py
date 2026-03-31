import pickle
import numpy as np
from pathlib import Path
from PIL import Image  # 需要导入 PIL 来保存图片
from tqdm import tqdm

data_path = 'output/dvgt2/visual/geometry/default/openscene/fix_40_future_8/openscene#log-0006-scene-0012#0#cliplen#40/saved_data.pkl'
# 使用 pathlib 构建路径更为优雅
output_dir = Path(data_path).parent / 'rgb_per_view'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

# images: (T, V, 3, H, W)，范围是 0~1
images = data['images'][0]

# 获取各个维度的大小
T, V, C, H, W = images.shape

print(f"开始保存图片，共 {V} 个视角，每个视角 {T} 帧...")

for v in tqdm(range(V)):
    # 为当前视角创建文件夹：view_0, view_1...
    view_dir = output_dir / f'view_{v}'
    view_dir.mkdir(parents=True, exist_ok=True)
    
    for t in range(T):
        # 提取单张图片，此时形状为 (3, H, W)
        img_array = images[t, v]
        
        # 将形状从 (C, H, W) 转换为 (H, W, C)
        img_array = np.transpose(img_array, (1, 2, 0))
        
        # 将 [0, 1] 的浮点数转换为 [0, 255] 的无符号整型
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        
        # 转换为 PIL Image 对象并保存
        img = Image.fromarray(img_array)
        save_path = view_dir / f'frame_{t:04d}.png'
        img.save(save_path)

print(f"✅ 所有图片已成功保存至: {output_dir}")