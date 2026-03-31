import pandas as pd

# --- dvgt 1 openscene test sample 200---
# 输入的 Parquet 文件路径
INPUT_PARQUET_PATH = 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/openscene_test.parquet'
# 输出的 Parquet 文件路径
OUTPUT_PARQUET_PATH = 'data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/openscene_test_sample_200.parquet'
# 场景所需的最少帧数
MIN_FRAMES_THRESHOLD = 40
# 需要采样的场景数量
NUM_SCENES_TO_SAMPLE = 200
# 随机种子，确保每次采样结果一致
RANDOM_SEED = 42

def filter_and_sample_scenes(
    input_path: str,
    output_path: str,
    min_frames: int,
    num_samples: int,
    random_seed: int
):
    """
    读取自驾数据集的 Parquet 文件，筛选并采样场景。

    处理流程:
    1. 读取 Parquet 文件到 pandas DataFrame。
    2. 按 'scene_id' 分组，计算每个场景的唯一帧数。
    3. 筛选出帧数 > min_frames 的场景。
    4. 从符合条件的场景中随机采样 num_samples 个场景。
    5. 提取这些被采样场景的所有数据。
    6. 将结果保存到新的 Parquet 文件。

    Args:
        input_path (str): 输入 Parquet 文件的路径。
        output_path (str): 输出 Parquet 文件的路径。
        min_frames (int): 场景所需的最少帧数。
        num_samples (int): 要采样的场景数量。
        random_seed (int): 用于采样的随机种子。
    """
    print(f"--- 开始处理 Parquet 文件 ---")
    
    # 1. 读取 Parquet 文件
    try:
        print(f"正在读取文件: {input_path}")
        df = pd.read_parquet(input_path)
        print("文件读取成功。数据信息如下:")
        df.info(memory_usage='deep')
    except FileNotFoundError:
        print(f"错误: 文件未找到 at {input_path}")
        return

    # 2. 按 'scene_id' 分组，计算每个场景的唯一帧数
    # 使用 nunique() 是因为同一帧(frame_idx)可能有多行数据（不同摄像头）
    print(f"\n按 'scene_id' 分组并计算唯一帧数...")
    frame_counts_per_scene = df.groupby('scene_id')['frame_idx'].nunique()
    
    # 3. 筛选出帧数 >= min_frames 的场景
    print(f"筛选帧数 >= {min_frames} 的场景...")
    eligible_scenes = frame_counts_per_scene[frame_counts_per_scene >= min_frames]
    
    num_eligible_scenes = len(eligible_scenes)
    print(f"找到 {num_eligible_scenes} 个符合条件的场景。")

    if num_eligible_scenes == 0:
        print("错误: 没有找到任何符合条件的场景，程序退出。")
        return

    # 4. 从符合条件的场景中随机采样
    eligible_scene_ids = eligible_scenes.index.tolist()

    if num_eligible_scenes < num_samples:
        print(f"警告: 符合条件的场景数量 ({num_eligible_scenes}) 少于期望采样数量 ({num_samples})。")
        print(f"将使用所有 {num_eligible_scenes} 个符合条件的场景。")
        sampled_scene_ids = eligible_scene_ids
    else:
        print(f"从 {num_eligible_scenes} 个场景中随机采样 {num_samples} 个...")
        # 使用 pandas.Series.sample 进行高效采样
        sampled_scene_ids = pd.Series(eligible_scene_ids).sample(
            n=num_samples, 
            random_state=random_seed
        ).tolist()
    
    print(f"已采样 {len(sampled_scene_ids)} 个场景ID。")

    # 5. 提取这些被采样场景的所有数据
    print("正在从原始数据中提取被采样场景的全部数据...")
    # 使用 isin() 方法可以非常高效地进行筛选
    sampled_df = df[df['scene_id'].isin(sampled_scene_ids)].copy()

    # 6. 将结果保存到新的 Parquet 文件
    try:
        print(f"正在将结果保存到: {output_path}")
        sampled_df.to_parquet(output_path, engine='pyarrow', compression='zstd', index=False)
        print("文件保存成功！")
        print(f"\n最终输出的数据包含 {sampled_df['scene_id'].nunique()} 个场景和 {len(sampled_df)} 行数据。")
    except Exception as e:
        print(f"错误: 保存文件失败。 {e}")
        
    print(f"--- 处理完成 ---")

if __name__ == '__main__':
    filter_and_sample_scenes(
        input_path=INPUT_PARQUET_PATH,
        output_path=OUTPUT_PARQUET_PATH,
        min_frames=MIN_FRAMES_THRESHOLD,
        num_samples=NUM_SCENES_TO_SAMPLE,
        random_seed=RANDOM_SEED
    )