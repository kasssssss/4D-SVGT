import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import os
import argparse
import torch
import datetime
import logging
from tqdm import tqdm
from dataloaders import get_dataset
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from typing import List, Dict, Any, Tuple, Callable
from collections import defaultdict
import numpy as np
import pandas as pd
import torch.nn as nn

from third_party.MoGe.moge.model.v2 import MoGeModel
from tools.data_process.utils.io import write_depth, write_image
from tools.data_process.utils.alignment import roe_align
from tools.data_process.utils.log import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--ckpt_path", type=str, default="ckpt/moge-2-vitl/model.pt")
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--num_workers", type=int, default=36)
    parser.add_argument('--save_workers', type=int, default=128, help="异步保存深度图和图片的总线程数")
    parser.add_argument('--gen_meta', action='store_true')
    parser.add_argument('--gen_proj_depth', action='store_true')
    parser.add_argument('--gen_pred_depth', action='store_true')
    parser.add_argument('--gen_align_depth', action='store_true')
    parser.add_argument('--gen_image', action='store_true')
    parser.add_argument("--meta_storage_path", default="data_annotation/meta/moge_v2_large_correct_focal")
    parser.add_argument("--split", default="train")
    parser.add_argument("--log_dir", default="output/data_process/label_data")
    return parser.parse_args()

def setup_distributed() -> Tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    return rank, world_size, device_id

def save_results_rank0(
    results: List[List[Tuple]],
    executor: ThreadPoolExecutor,
    write_fn: Callable,
    rank: int,
    world_size: int
) -> None:
    gathered_data = [None] * world_size if rank == 0 else None
    dist.gather_object(results, gathered_data, dst=0)
    if rank == 0:
        for rank_data in gathered_data:
            if rank_data:
                for save_path, file in rank_data:
                    executor.submit(write_fn, save_path, file)

def custom_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    用于根据 image_size 对批次中的样本进行分组和堆叠。
    返回: List[Dict[str, Any]]: 每个字典，代表一个具有相同 image_size 的子批次。
    """
    grouped_samples = defaultdict(list)
    for sample in batch:
        grouped_samples[sample['image_size']].append(sample)

    processed_batches = []
    for image_size, samples in grouped_samples.items():
        collated_group = defaultdict(list)
        for sample in samples:
            for key, value in sample.items():
                collated_group[key].append(value)

        final_group = {}
        for key, values in collated_group.items():
            first_item = values[0]
            if isinstance(first_item, torch.Tensor):
                final_group[key] = torch.stack(values)
            elif isinstance(first_item, np.ndarray):
                final_group[key] = np.stack(values)
            else:   # 其他类型，收集为List
                final_group[key] = values
        processed_batches.append(final_group)
    return processed_batches

def process_pred_depth_batch(batch: Dict[str, Any], model: MoGeModel, device_id: int) -> List[Tuple[str, np.ndarray]]:
    input_images = batch['image_input'].to(device=device_id, non_blocking=True)
    input_images = input_images.permute(0, 3, 1, 2) / 255
    
    model_output = model.infer(input_images, fov_x=batch['fov'], use_fp16=False, force_projection=False)
    pred_depth = model_output['depth']   # mask的地方已经被设为INF了
    # resize使得fx != fy
    pred_depth = nn.functional.interpolate(pred_depth.unsqueeze(1), size=batch['image_save'].shape[1: 3], mode='nearest').squeeze(1)

    # 添加到batch，后续align depth可能会使用
    batch['pred_depth'] = pred_depth

    pred_depth = pred_depth.cpu().numpy()

    return list(zip(
        batch['save_pred_depth_path'],
        pred_depth,
    ))

def process_align_depth_batch(batch: Dict[str, Any], device_id: int) -> List[Tuple[str, np.ndarray]]:
    proj_depth = batch['proj_depth'].to(device_id, non_blocking=True)
    pred_depth = batch['pred_depth'].to(device_id, non_blocking=True)

    # align
    aligned_depth, scale, shift = roe_align(pred_depth, proj_depth, batch['save_align_depth_path'])
    aligned_depth = aligned_depth.cpu().numpy()

    return list(zip(
        batch['save_align_depth_path'],
        aligned_depth,
    ))

def process_image_batch(batch: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    return list(zip(
        batch['save_image_path'],
        batch['image_save']
    ))

def process_proj_depth_batch(batch: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    return list(zip(
        batch['save_proj_depth_path'],
        batch['proj_depth'].numpy(),
    ))

def process_meta_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 将Dict[List[item]]拆分为：List[Dict[item]]
    keys = {
        'scene_id', 'frame_idx', 'cam_type', 'filename', 'intrinsics', 'extrinsics', 'ego_to_world',
        'ego_velocity', 'ego_acceleration', 'driving_command', 
        'frame_token',
    }
    # 'ego_velocity', 'ego_acceleration', 'driving_command', 'frame_token'，这几个有的数据集没有
    keys = keys.intersection(batch.keys())
    return [dict(zip(keys, values)) for values in zip(*(batch[key] for key in keys))]

def main() -> None:
    args = parse_args()

    rank, world_size, device_id = setup_distributed()
    current_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
    setup_logger(log_name=f"{current_time}", rank=rank, log_dir=os.path.join(args.log_dir, args.dataset_name, args.split))

    if not (args.gen_proj_depth or args.gen_pred_depth or args.gen_meta or args.gen_image or args.gen_align_depth):
        if rank == 0:
            logging.warning("未指定 --gen_proj_depth, --gen_pred_depth, --gen_image, -gen_align_depth 或 --gen_meta，程序将不执行任何操作并退出。")
        return
    logging.info(f"gen_meta: {args.gen_meta}, gen_proj_depth: {args.gen_proj_depth}, gen_pred_depth: {args.gen_pred_depth}, gen_align_depth: {args.gen_align_depth}, gen_image: {args.gen_image}")
    
    dataset = get_dataset(args.dataset_name, split=args.split)
    logging.info("successful loading dataset!")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
        collate_fn=custom_collate_fn
    )

    # 模型 & 多进程存储depth
    model = None
    save_executor = None
    if args.gen_pred_depth:
        model = MoGeModel.from_pretrained(args.ckpt_path).to(device_id).eval()
    if rank == 0:
        # 这里必须汇集到单卡执行写入，我们4机32卡，bs=32，如果多卡多进程写入一次最大1024写入，程序无法处理
        save_executor = ThreadPoolExecutor(max_workers=args.save_workers, thread_name_prefix='save_depth') if rank == 0 else None

    all_metas = []
    with torch.no_grad():
        for sub_batches in tqdm(data_loader, desc=f"Processing {args.dataset_name}", disable=(rank != 0)):
            images_this_iter = []
            pred_depth_results_this_iter = []
            proj_depth_results_this_iter = []
            align_depth_results_this_iter = []
            meta_results_this_iter = []

            for batch in sub_batches:
                if args.gen_image:
                    images_this_iter.extend(process_image_batch(batch))
                if args.gen_pred_depth:
                    pred_depth_results_this_iter.extend(process_pred_depth_batch(batch, model, device_id))
                if args.gen_meta:
                    meta_results_this_iter.extend(process_meta_batch(batch))
                if args.gen_proj_depth:
                    proj_depth_results_this_iter.extend(process_proj_depth_batch(batch))
                if args.gen_align_depth:
                    align_depth_results_this_iter.extend(process_align_depth_batch(batch, device_id))

            # 使用rank 0 保存文件，防止多线程打架
            # 这里需要统一收集后传递数据，防止不同rank 小batch数量不同，单个循环时一直等待超时
            if args.gen_pred_depth:
                save_results_rank0(pred_depth_results_this_iter, save_executor, write_depth, rank, world_size)
                    
            if args.gen_proj_depth:
                save_results_rank0(proj_depth_results_this_iter, save_executor, write_depth, rank, world_size)

            if args.gen_align_depth:
                save_results_rank0(align_depth_results_this_iter, save_executor, write_depth, rank, world_size)

            if args.gen_image:
                save_results_rank0(images_this_iter, save_executor, write_image, rank, world_size)
                
            if args.gen_meta:
                gathered_metas = [None] * world_size if rank == 0 else None
                dist.gather_object(meta_results_this_iter, gathered_metas, dst=0)
                if rank == 0:
                    for rank_metas in gathered_metas:
                        if rank_metas:
                            all_metas.extend(rank_metas)

            dist.barrier()

    # 确保不会有进程提前退出导致整个 job 被 kill
    dist.barrier()
    if rank == 0:
        if args.gen_pred_depth or args.gen_proj_depth or args.gen_align_depth or args.gen_image:
            save_executor.shutdown(wait=True)
            logging.info("处理完成。")

        if args.gen_meta:
            df = pd.DataFrame(all_metas)

            # 使用 'filename' 作为唯一标识符进行去重，因为DistributedSampler设置的batch size不能整除整个数据集，会导致一些文件重复
            logging.info(f"去重前元数据条数: {len(df)}")
            df.drop_duplicates(subset=['filename'], inplace=True)
            logging.info(f"去重后元数据条数: {len(df)}")

            output_path = os.path.join(args.meta_storage_path, f"{args.dataset_name}_{args.split}.parquet")

            os.makedirs(args.meta_storage_path, exist_ok=True)
            df.to_parquet(output_path, engine='pyarrow', compression='zstd', index=False)
            logging.info(f"元数据已保存至: {output_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()