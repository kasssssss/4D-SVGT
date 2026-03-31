import sys
sys.path.append("path/to/sam3")
import os
import glob
import cv2
from sam3.model_builder import build_sam3_video_predictor
import argparse
import numpy as np

parser = argparse.ArgumentParser()
# 注意：这里默认路径修改为了包含所有 view 的父目录
parser.add_argument("--pth", type=str, \
    default="output/dvgt2/visual/geometry/default/openscene/fix_40_future_8/openscene#log-0006-scene-0012#0#cliplen#40/rgb_per_view")
args = parser.parse_args()

# ================= 1. 初始化模型 =================
video_predictor = build_sam3_video_predictor(
    checkpoint_path='/mnt/ad-alg/planning-users/xiezixun/public_checkpoints/sam3/sam3.pt',
    gpus_to_use=[0] # 根据你的 nvidia-smi，指定使用 GPU 0
)

# ================= 2. 定义处理函数 =================
def propagate_in_video(predictor, session_id):
    # propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def track_class(predictor, v_path, prompt_text):
    """
    针对某一个类别（如 "person"）进行全局追踪。
    """
    print(f"---> 开始追踪类别: {prompt_text}")
    response = predictor.handle_request(request=dict(type="start_session", resource_path=v_path))
    session_id = response["session_id"]
    
    predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt_text,
        )
    )

    outputs_per_frame = propagate_in_video(predictor, session_id)

    # 跑完后关闭 session，释放该段视频产生的缓存
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return outputs_per_frame


# ================= 3. 遍历视角目录并进行推理 =================
base_path = args.pth

# 找到当前目录下所有的子文件夹（视角）并排序
view_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

if not view_dirs:
    print(f"在 {base_path} 目录下没有找到任何子目录，请检查路径。")

for view_dir in view_dirs:
    print(f"\n================= 正在处理视角: {view_dir} =================")
    
    video_path = os.path.join(base_path, view_dir)
    # 替换路径里的 rgb 为 seg，比如 rgb_per_view -> seg_per_view
    output_dir = video_path.replace('rgb', 'seg')
    os.makedirs(output_dir, exist_ok=True)

    # 获取当前视角下的所有图片
    img_files = glob.glob(os.path.join(video_path, "*.[jp][pn]g")) # 兼容 jpg 和 png
    if not img_files:
        print(f"警告：视角目录 {video_path} 中没有找到图片，跳过该目录。")
        continue
        
    img_files.sort()
    # 读一张图获取高宽，用于兜底生成空白 mask
    sample_img = cv2.imread(img_files[0])
    H, W = sample_img.shape[:2]

    # 分别追踪人、车、包
    person_masks_dict = track_class(video_predictor, video_path, "pedestrian")
    car_masks_dict = track_class(video_predictor, video_path, "vehicle")
    handbag_dict = track_class(video_predictor, video_path, "handbag")

    # 合并 Mask 并保存
    print(f"---> 开始合并并保存 {view_dir} 的 Mask 图像...")
    for frame_idx in person_masks_dict.keys():
        mask_person = person_masks_dict[frame_idx]['out_binary_masks']
        mask_car = car_masks_dict[frame_idx]['out_binary_masks']
        mask_bag = handbag_dict[frame_idx]['out_binary_masks']

        mask_person = np.any(mask_person, axis=0) if mask_person.shape[0] > 0 else np.zeros((H, W), dtype=bool)
        mask_car = np.any(mask_car, axis=0) if mask_car.shape[0] > 0 else np.zeros((H, W), dtype=bool)
        mask_bag = np.any(mask_bag, axis=0) if mask_bag.shape[0] > 0 else np.zeros((H, W), dtype=bool)

        # 取并集 (Bitwise OR)
        final_mask_bool = mask_person | mask_car | mask_bag
        
        # 转换为 0 或 255 的灰度图格式
        final_mask_img = (final_mask_bool.astype(np.uint8)) * 255
        
        # 提取原本图片的文件名
        orig_basename = os.path.basename(img_files[frame_idx])
        name_without_ext = os.path.splitext(orig_basename)[0]
        save_name = f"{name_without_ext}.png"
        
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, final_mask_img)

    print(f"完成！{view_dir} 的所有 Mask 均已保存至: {output_dir}")

# ================= 4. 所有推理结束，关闭模型 =================
video_predictor.shutdown()
print("\n所有视角均已处理完毕，Predictor已关闭。")