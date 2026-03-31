import sys
sys.path.append('/mnt/ad-alg/planning-users/xiezixun/vggt_work_dir/sam3')

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["SAM3_COLLECTIVE_OP_TIMEOUT_SEC"] = "1800"
os.environ["NCCL_TIMEOUT"] = "1800000"

import argparse
import logging
import json
import tempfile
import cv2
import numpy as np
import pandas as pd
import torch
import uuid
import pickle
from pathlib import Path

from sam3.model_builder import build_sam3_video_predictor

def safe_padding_img_path(frame_mapping, pad_to_divisor, temp_dir):
    # 确保文件数量能被 GPU 数量整除，避免sam内部卡死
    orig_frame_count = len(frame_mapping)
    if pad_to_divisor > 0:
        remainder = orig_frame_count % pad_to_divisor
        if remainder != 0:
            pad_count = pad_to_divisor - remainder
            # 找到最后一帧的原始路径，用作 Padding 的填充画面
            last_orig_path = frame_mapping[orig_frame_count - 1]['orig_path']
            last_orig_path = os.path.abspath(last_orig_path)
            
            # 继续向后创建软链接，补充缺失的帧
            for p in range(pad_count):
                pad_idx = orig_frame_count + p
                symlink_name = f"{pad_idx:05d}.jpg"
                symlink_path = os.path.join(temp_dir, symlink_name)
                os.symlink(last_orig_path, symlink_path)
            # 注意：我们故意不把这些 pad_idx 加入 frame_mapping，这样后续自然就不会去读取它们的结果

def propagate_in_video(predictor, session_id):
    """从第0帧向后传播追踪"""
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
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    
    return outputs_per_frame

def process_single_view(predictor, view_df, img_save_root, seg_save_root, pad_to_divisor=8):
    """处理单个场景中的单个视角的所有帧"""
    # 按帧序号排序，注意这里移除了 reset_index，为了保留原 dataframe 的 index 以便最后 merge
    view_df = view_df.sort_values(by='frame_idx')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_mapping = {}  
        uuid_map = {} # 记录: (类别, 局部id) -> 全局唯一 UUID 字符串
        track_results_for_view = {} # 记录: 原 df_index -> 本帧 track 信息
        
        # 为了保证 SAM3 的读取顺序，使用枚举 i 来生成软链接
        for i, (orig_idx, row) in enumerate(view_df.iterrows()):
            orig_img_path = os.path.join(img_save_root, row['filename']) + '.jpg'
            abs_orig_img_path = os.path.abspath(orig_img_path)
                
            symlink_name = f"{i:05d}.jpg"
            symlink_path = os.path.join(temp_dir, symlink_name)
            os.symlink(abs_orig_img_path, symlink_path)
            
            frame_mapping[i] = {
                'orig_index': orig_idx,
                'orig_path': orig_img_path,
                'filename': row['filename']
            }

        safe_padding_img_path(frame_mapping, pad_to_divisor, temp_dir)

        # 分别追踪两类物体
        vehicle_masks_dict = track_class(predictor, temp_dir, "vehicle")
        pedestrian_masks_dict = track_class(predictor, temp_dir, "pedestrian")
        
        # 结果融合与保存
        for sam_frame_idx, mapping in frame_mapping.items():
            orig_idx = mapping['orig_index']
            orig_path = mapping['orig_path']
            filename = mapping['filename']
            
            # 读取原图获取目标分辨率
            orig_img = cv2.imread(orig_path)
            if orig_img is None:
                continue
            orig_h, orig_w = orig_img.shape[:2]
            
            final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            frame_track_info = []
            
            veh_data = vehicle_masks_dict.get(sam_frame_idx, {})
            ped_data = pedestrian_masks_dict.get(sam_frame_idx, {})
            
            class_configs = [(2, ped_data), (1, veh_data)]
            
            for cls_val, cls_data in class_configs:
                if not cls_data or len(cls_data.get('out_obj_ids', [])) == 0:
                    continue
                    
                masks = cls_data['out_binary_masks']  # (N, H, W) 已经和原图一样大
                boxes = cls_data['out_boxes_xywh']    # (N, 4)
                probs = cls_data['out_probs']         # (N,)
                obj_ids = cls_data['out_obj_ids']     # (N,)
                
                cls_mask_merged = np.any(masks, axis=0).astype(np.uint8)
                
                # 将该类别写入最终 Mask (后执行的车会覆盖先执行的人)
                final_mask[cls_mask_merged > 0] = cls_val
                
                for j in range(len(obj_ids)):
                    local_id = int(obj_ids[j])
                    unique_key = (cls_val, local_id)
                    
                    if unique_key not in uuid_map:
                        uuid_map[unique_key] = uuid.uuid4().hex
                        
                    frame_track_info.append({
                        "uuid": uuid_map[unique_key],
                        "class": cls_val,
                        "xywh": boxes[j].tolist(),
                        "probs": float(probs[j])
                    })

            # 构建保存路径
            seg_out_path = os.path.join(seg_save_root, filename)
            seg_img_path = os.path.splitext(seg_out_path)[0] + '.png'
            
            os.makedirs(os.path.dirname(seg_img_path), exist_ok=True)
            
            # 保存 Seg 结果
            cv2.imwrite(seg_img_path, final_mask)
            
            # 记录 Track 结果，准备返回给主函数
            track_results_for_view[orig_idx] = json.dumps(frame_track_info)

    return track_results_for_view

"""
conda activate sam3
python tools/data_process/sam3_anotation.py \
    --parquet_path=data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/openscene_trainval_slice_1.parquet
"""

def main():
    parser = argparse.ArgumentParser(description="SAM3 OpenScene Auto Annotation")
    parser.add_argument("--parquet_path", type=str, default="data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/openscene_trainval_slice_1.parquet")
    parser.add_argument("--seg_parquet_save_root", type=str, default="data_annotation/meta/moge_v2_large_correct_focal/moge_pred_filter_parquet/seg")
    parser.add_argument("--img_save_root", type=str, default="data_annotation/image/moge_v2_large_correct_focal")
    parser.add_argument("--seg_save_root", type=str, default="data_annotation/segmentation/moge_v2_large_correct_focal")
    parser.add_argument("--ckpt", type=str, default="/mnt/ad-alg/planning-users/xiezixun/public_checkpoints/sam3/sam3.pt", help="SAM3 模型权重路径")
    
    args = parser.parse_args()

    dataset_name = os.path.basename(args.parquet_path).split('_')[0]
    args.img_save_root = os.path.join(args.img_save_root, dataset_name)
    args.seg_save_root = os.path.join(args.seg_save_root, dataset_name)
    
    os.makedirs(args.seg_parquet_save_root, exist_ok=True)
        
    print("=> 正在读取 Parquet 数据...")
    df = pd.read_parquet(args.parquet_path)
    
    num_gpus = torch.cuda.device_count()
    gpus_to_use = list(range(num_gpus))
    print(f"=> 初始化 SAM3 多卡模型，使用 GPU: {gpus_to_use} ...")
    
    video_predictor = build_sam3_video_predictor(
        checkpoint_path=args.ckpt,
        gpus_to_use=gpus_to_use
    )

    # 处理日志
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(logging.WARNING)
            
    grouped_scenes = df.groupby('scene_id')
    total_scenes = len(grouped_scenes)
    
    # ==================== resume ====================
    resume_file = Path(args.seg_parquet_save_root) / f"{Path(args.parquet_path).stem}_resume.pkl"
    if resume_file.exists():
        print(f"\n=> [Resume] 发现中断记录，正在从 {resume_file} 恢复进度...")
        with open(resume_file, 'rb') as f:
            state = pickle.load(f)
            completed_scenes = state.get("completed_scenes", set())
            all_track_results = state.get("track_results", {})
        print(f"=> [Resume] 成功恢复，已完成 {len(completed_scenes)} 个场景！\n")
    else:
        completed_scenes = set()
        all_track_results = {}
    # ====================================================================
    
    for idx, (scene_id, scene_df) in enumerate(grouped_scenes, 1):       
        progress_pct = (idx / total_scenes) * 100

        # 👉 新增：如果该 scene 已存在于完成记录中，直接跳过
        if scene_id in completed_scenes:
            print(f"⏩ [总体进度] {idx} / {total_scenes} | 场景 {scene_id} 已在历史记录中，跳过。")
            continue

        # 打印极其醒目的进度块，防止被底层的 tqdm 覆盖
        print("\n" + "=" * 60)
        print(f"🚀 [总体进度] {idx} / {total_scenes} ({progress_pct:.1f}%) | 当前场景: {scene_id}, 帧：{len(scene_df) // 8}")
        print("=" * 60 + "\n")

        grouped_views = scene_df.groupby('cam_type')
        
        for view_name, view_df in grouped_views:
            view_tracks = process_single_view(video_predictor, view_df, args.img_save_root, args.seg_save_root)
            all_track_results.update(view_tracks)

        # ==================== resume ====================
        completed_scenes.add(scene_id)
        # 每次跑完一个 scene 就存一下，使用 pickle 保存极快，几乎不影响耗时
        with open(resume_file, 'wb') as f:
            pickle.dump({
                "completed_scenes": completed_scenes,
                "track_results": all_track_results
            }, f)
        # ===============================================================

    print("\n=> 所有数据标注完成！正在关闭模型...")
    video_predictor.shutdown()
    del video_predictor

    print("=> 正在将 Track 结果追加到 Parquet 文件中...")
    track_series = pd.Series(all_track_results)
    df['track_info'] = track_series

    out_parquet_path = Path(args.seg_parquet_save_root) / Path(args.parquet_path).name
    df.to_parquet(out_parquet_path, engine='pyarrow', compression='zstd', index=False)

    print(f"=> 全部完成！新的包含追踪信息的 Parquet 已保存至:\n   {out_parquet_path}")

    # 👉 新增：大功告成后清理临时缓存文件
    if resume_file.exists():
        os.remove(resume_file)
        print(f"=> 已自动清理进度缓存文件: {resume_file.name}")

if __name__ == "__main__":
    main()