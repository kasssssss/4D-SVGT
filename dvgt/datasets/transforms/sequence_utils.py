import random
import logging
import pandas as pd
from typing import Dict, List, Tuple, Any
import yaml
from tqdm import tqdm

def split_scenes_into_clips(
    scenes: Dict[str, pd.DataFrame], 
    clip_len: int, 
    fps_step: int, 
    dataset_name: str,
    drop_single_frame: bool = False,
    only_first_clip: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    将场景字典中的长场景切分为指定长度的短片段。

    Args:
        scenes (Dict[str, pd.DataFrame]): 
            一个字典，键是 scene_id，值是对应的 DataFrame。
            每个 DataFrame 的索引是 ['frame_idx', 'cam_type']。
        clip_len (int): 
            每个片段的最大帧数。
        fps_step (int):
            每帧的采样步长。
        drop_single_frame (bool):
            如果为 True，且切分后的片段（或原场景）仅包含 1 个时间戳（unique frame），则丢弃该片段。
        only_first_clip (bool):
            如果为 True，则每个场景只取第一个片段
    Returns:
        Dict[str, pd.DataFrame]: 
            一个新的字典，其中长场景被切分为多个短片段，并以 f"{scene_id}#N" 的格式命名。
    """
    new_scenes: Dict[str, pd.DataFrame] = {}

    for scene_id, scene_df in scenes.items():             
        unique_frames = scene_df.index.get_level_values('frame_idx').unique().sort_values()
        
        sampled_frames = unique_frames[::fps_step]
        num_sampled = len(sampled_frames)
        
        clip_count = 0
        for i in range(0, num_sampled, clip_len):
            clip_frame_indices = sampled_frames[i : i + clip_len]
            
            # 检查是否需要丢弃单帧片段
            if drop_single_frame and len(clip_frame_indices) == 1:
                continue
            
            # 检查是否只需要第一个片段
            if only_first_clip and i > 0:
                continue

            clip_df = scene_df.loc[clip_frame_indices].copy()
            
            new_scene_id = f"{dataset_name}#{scene_id}#{clip_count}#cliplen#{len(clip_frame_indices)}"
            new_scenes[new_scene_id] = clip_df
            
            clip_count += 1
                
    return new_scenes

def split_scenes_into_windows(
    scenes: Dict[str, pd.DataFrame], 
    scene_window: int, 
    fps_step: int,
    dataset_name: str,
) -> Dict[str, pd.DataFrame]:
    """
    将场景切为按照滑动窗口滚动向前的片段。
    比如一个400帧的场景，窗口大小是10，fps_step=5
    - 先按照fps_step采样，得到80帧场景
    - 0-9帧为一个片段，1-10帧是一个片段，以此类推，70-79帧是一个片段

    Args:
        scenes (Dict[str, pd.DataFrame]): 
            一个字典，键是 scene_id，值是对应的 DataFrame。
            每个 DataFrame 的索引是 ['frame_idx', 'cam_type']。
        scene_window (int): 
            每个片段的窗口。
        fps_step (int):
            每帧的采样步长。

    Returns:
        Dict[str, pd.DataFrame]: 
            一个新的字典，其中长场景被切分为多个短片段，并以 f"{scene_id}#N" 的格式命名。
    """
    
    result_scenes: Dict[str, pd.DataFrame] = {}

    for scene_id, df in scenes.items():
        all_frames = df.index.get_level_values('frame_idx').unique().sort_values()

        # 按照 fps_step 进行采样
        sampled_frames = all_frames[::fps_step]

        # 如果采样后的帧数不足以构成一个窗口，跳过该场景
        num_sampled = len(sampled_frames)
        if num_sampled < scene_window:
            continue

        # 计算可以生成的窗口数量
        # 例如：11帧，窗口10，步长1 -> (11 - 10) + 1 = 2 个窗口
        num_windows = num_sampled - scene_window + 1

        for i in range(num_windows):
            current_window_frames = sampled_frames[i : i + scene_window]
            sub_df = df.loc[current_window_frames].copy()            
            new_key = f"{dataset_name}#{scene_id}#{i}#window#{scene_window}"
            result_scenes[new_key] = sub_df

    return result_scenes

def get_nuscenes_logs(
    scenes: Dict[str, pd.DataFrame], 
    fps_step: int = 1, 
    num_history_frames: int = 4,
    future_frames: int = 6
) -> Dict[str, pd.DataFrame]:    

    scene_window = num_history_frames + future_frames

    result_scenes: Dict[str, pd.DataFrame] = {} 

    for scene_id, df in scenes.items():
        all_frames = df.index.get_level_values('frame_idx').unique().sort_values()

        # 按照 fps_step 进行采样
        sampled_frames = all_frames[::fps_step]

        # 如果采样后的帧数不足以构成一个窗口，跳过该场景
        num_sampled = len(sampled_frames)
        if num_sampled < scene_window:
            continue

        # 计算可以生成的窗口数量
        # 例如：11帧，窗口10，步长1 -> (11 - 10) + 1 = 2 个窗口
        num_windows = num_sampled - scene_window + 1

        for i in range(num_windows):
            current_window_frames = sampled_frames[i : i + scene_window]
            sub_df = df.loc[current_window_frames].copy()

            target_frame_idx = current_window_frames[num_history_frames - 1]
            mask = sub_df.index.get_level_values('frame_idx') == target_frame_idx
            new_key = sub_df[mask].iloc[0]['frame_token']

            result_scenes[new_key] = sub_df

    return result_scenes

def sample_training_sequence(
    sequence_list: List[str],
    scenes: Dict[str, pd.DataFrame],
    fps_step: int,
    num_frames: int,
    seq_index: int,
    future_frame_num: int = 0,
    inside_random: bool = True,
) -> Tuple[str, pd.DataFrame, List[int], List[int]]:
    """
    在训练过程中，随机寻找一个足够长的 Sequence。
    
    Returns:
        tuple: (seq_name, scene_df, sampled_frame_ids, future_frame_ids)
    """
    # 帧采样，我们约定所有场景的帧都应该连续
    required_frames_span = num_frames + future_frame_num

    # 避免有些场景的帧数不够
    while True:
        if inside_random:
            seq_index = random.randint(0, len(sequence_list) - 1)       # 左闭右闭

        seq_name = sequence_list[seq_index]
        scene_df = scenes[seq_name]
        all_frames = scene_df.index.get_level_values('frame_idx').unique().sort_values()

        # 我们在这里消除fps_step的影响，后续将直接取消fps这个设定
        available_frames = all_frames[::fps_step]

        if len(available_frames) >= required_frames_span:
            max_start = len(available_frames) - required_frames_span
            start_idx = random.randint(0, max_start)
            # 注意场景available_frames不一定从0开始
            # 注意这里sampled_frame_ids，不包含未来帧
            sampled_frame_ids = available_frames[start_idx: start_idx + num_frames]
            future_frame_ids = []
            if future_frame_num > 0:
                future_frame_ids = available_frames[start_idx + num_frames : start_idx + num_frames + future_frame_num]
            return seq_name, scene_df, sampled_frame_ids, future_frame_ids

        else:   # 重新选择一个 scene
            logging.debug(
                f"Sequence {seq_name} has not enough frames. "
                f"Required span: {required_frames_span}, Available: {len(available_frames)}. Retrying..."
            )
            if not inside_random:  # WARNING inside_random=False的情况需要测试
                seq_index = (seq_index + 1) % len(sequence_list)


def get_navtest_logs(
    scenes: Dict[str, pd.DataFrame], 
    config_path: str = "third_party/navsim_v1_1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
    num_history_frames: int = 4,
    frame_interval: int = 1,
    future_frames: int = 8
) -> Dict[str, pd.DataFrame]:
    
    # 解析 YAML 配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
            
    past_frames = num_history_frames - 1  # 3帧
    valid_tokens = set(config.get('tokens', []))
    
    token_to_scene_loc = {}
    
    # ==========================================
    # 1. 扫描 Scenes，建立 Token 映射并统计视角数量
    # ==========================================
    for scene_id, df_scene in tqdm(scenes.items()):
        # 快速过滤出包含目标 token 的行
        mask = df_scene['frame_token'].isin(valid_tokens)
        if not mask.any():
            continue
            
        valid_df = df_scene[mask]
                    
        # 提取 token 所在的 scene_id 和 对应的 frame_idx
        # 通过 reset_index 可以将 MultiIndex 释放出来方便读取 frame_idx
        unique_mapping = valid_df.reset_index().drop_duplicates(subset=['frame_token'])
        for _, row in unique_mapping.iterrows():
            token_to_scene_loc[row['frame_token']] = (scene_id, row['frame_idx'])

    # 提取 Clip 片段
    new_scenes: Dict[str, pd.DataFrame] = {}
    not_continuous_count = 0

    for tok in valid_tokens:
            
        scene_id, center_frame_idx = token_to_scene_loc[tok]
        df_scene = scenes[scene_id]
        
        # 生成需要保留的目标帧索引列表
        # 考虑了 config 中的 frame_interval 步长
        target_frame_indices = [
            center_frame_idx + i * frame_interval 
            for i in range(-past_frames, future_frames + 1)
        ]
        
        # 针对 MultiIndex 的 level_0 ('frame_idx') 进行 isin 筛选
        idx_level_0 = df_scene.index.get_level_values('frame_idx')
        clip_df = df_scene[idx_level_0.isin(target_frame_indices)].copy()

        if len(clip_df) != (8 * (num_history_frames + future_frames)):       # 8视角，12帧
            not_continuous_count += 1
            logging.warning(f"lack frame!!! expect {8 * (num_history_frames + future_frames)}, get {len(clip_df)}, scene_id: {scene_id}, center_frame_idx: {center_frame_idx},\n \
                lack frame: {set(target_frame_indices) - set(clip_df.index.get_level_values('frame_idx').unique().sort_values())}")
            continue
        
        new_scenes[tok] = clip_df
    
    logging.warning(f"not_continuous_count: {not_continuous_count}")
        
    return new_scenes