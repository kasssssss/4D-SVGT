import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import argparse
import logging
import os
from typing import Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from tools.data_process.utils.io import read_depth
from tools.data_process.utils.alignment import roe_align
from dvgt.evaluation.metrics.depth_metrics import depth_evaluation_numpy

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=1, help="currently only support bs=1")
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--meta_storage_path", default="data_annotation/meta/moge_v2_large_correct_focal")
    parser.add_argument("--storage_pred_depth_path", default="data_annotation/pred_depth/moge_v2_large_correct_focal")
    parser.add_argument("--storage_proj_depth_path", default="data_annotation/proj_depth/moge_v2_large_correct_focal")
    parser.add_argument("--log_dir", default="output/data_process/filter_data/moge_pred_depth")
    parser.add_argument('--eval_align_depth', action='store_true', help="eval scale and shift for align depth")
    return parser.parse_args()

def setup_distributed() -> Tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    return rank, world_size, device_id

class DepthDataset(Dataset):
    def __init__(
        self, 
        parquet_path,
        dataset_name: str,
        storage_pred_depth_path: str = 'public_datasets/data_annotation/pred_depth/moge_v2_large',
        storage_proj_depth_path: str = 'public_datasets/data_annotation/proj_depth/moge_v2_large_correct_focal',
    ):
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        self.df = self.df.sort_values(by=['scene_id', 'cam_type', 'frame_idx']).reset_index(drop=True)
        self.pred_depth_path = Path(storage_pred_depth_path) / dataset_name
        self.proj_depth_path = Path(storage_proj_depth_path) / dataset_name

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _compute_stats(depth: np.ndarray) -> Dict[str, Any]:
        """Computes statistics for a given depth map."""
        valid_values = depth[np.isfinite(depth) & (depth > 0)]
        if valid_values.size == 0:
            return {"max": np.inf, "min": np.inf, "mean": np.inf, "valid_num": 0}
        
        return {
            "max": valid_values.max(),
            "min": valid_values.min(),
            "mean": valid_values.mean(),
            "valid_num": len(valid_values)
        }

    @staticmethod
    def _resize_depth_to_512(proj_depth: np.ndarray, pred_depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 将proj depth W resize到512，H等比缩小，但要求能被16整除，所以可能不会保持宽高比
        # pred depth直接reshape到对应大小
        original_h, original_w = proj_depth.shape
        scale_factor = 512 / original_w
        new_h, new_w = (int(round(original_h * scale_factor)), int(round(original_w * scale_factor)))

        # 计算能被 patch_size 整除的新的目标宽高
        patch_size = 16
        target_w = (new_w // patch_size) * patch_size
        target_h = (new_h // patch_size) * patch_size      

        proj_depth = cv2.resize(
            proj_depth,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

        pred_depth = cv2.resize(
            pred_depth,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return proj_depth, pred_depth

    @staticmethod
    def _calculate_depth_std(depth: np.ndarray) -> Dict:
        mask = np.isfinite(depth) & (depth > 0)
        if mask.sum() == 0:
            return {"depth_uv_std": 0, "depth_value_std": 0}
        depth_value_std = depth[mask].std()

        # 获取所有有效深度点的 (v, u) 坐标
        v_coords, u_coords = np.where(mask)
        v_std = v_coords.std()
        u_std = u_coords.std()
        depth_uv_std = (v_std + u_std) / 2.0
        
        return {"depth_uv_std": depth_uv_std, "depth_value_std": depth_value_std}

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data_dict = {}
        
        filename = Path(row['filename'])
        pred_depth_file = self.pred_depth_path / filename.with_suffix('.png')
        proj_depth_file = self.proj_depth_path / filename.with_suffix('.png')

        pred_depth = read_depth(pred_depth_file)[0].astype(np.float32)
        proj_depth = read_depth(proj_depth_file)[0].astype(np.float32)

        # 约定无效深度是0
        proj_depth_mask = np.isfinite(proj_depth) & (proj_depth > 0) & (proj_depth < 100)       # 额外限制gt的有效值小于100
        proj_depth[~proj_depth_mask] = 0
        pred_depth_mask = np.isfinite(pred_depth) & (pred_depth > 0)
        pred_depth[~pred_depth_mask] = 0

        # 我们约定将在宽为512的尺寸下，比较pred和proj depth，计算指标，因为这是我们训练时的尺寸
        proj_depth, pred_depth = self._resize_depth_to_512(proj_depth, pred_depth)

        # 计算一些统计量
        pred_stats = self._compute_stats(pred_depth)
        proj_stats = self._compute_stats(proj_depth)
        proj_depth_std = self._calculate_depth_std(proj_depth)

        data_dict = {
            # --- For Metrics Calculation ---
            "pred_depth": pred_depth,
            "proj_depth": proj_depth,

            # --- Metadata ---
            "scene_id": str(row['scene_id']),
            "frame_idx": row['frame_idx'],
            "cam_type": str(row['cam_type']),
            "filename": str(row['filename']),
            
            # --- Statistics for Analysis ---
            "pred_depth_valid_num": pred_stats["valid_num"],
            "proj_depth_valid_num": proj_stats["valid_num"],
            "pred_depth_max": pred_stats["max"],
            "pred_depth_min": pred_stats["min"],
            "pred_depth_mean": pred_stats["mean"],
            "proj_depth_max": proj_stats["max"],
            "proj_depth_min": proj_stats["min"],
            "proj_depth_mean": proj_stats["mean"],
            "proj_depth_uv_std": proj_depth_std["depth_uv_std"],
            "proj_depth_value_std": proj_depth_std["depth_value_std"],
        }

        return data_dict

def main():
    args = parse_args()
    rank, world_size, device_id = setup_distributed()

    assert args.batch_size == 1, "Evaluation currently only supports batch_size=1."

    output_dir = Path(args.log_dir) / args.dataset_name / args.split
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(args.meta_storage_path) / f"{args.dataset_name}_{args.split}.parquet"

    if rank == 0:
        print(f"Loading metadata from: {parquet_path}")
        print(f"Evaluating {'raw predicted depth' if args.eval_align_depth else 'pre-aligned depth'}.")
    
    dataset = DepthDataset(
        parquet_path, args.dataset_name, 
        storage_pred_depth_path=args.storage_pred_depth_path,
        storage_proj_depth_path=args.storage_proj_depth_path
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
        collate_fn=lambda x: x[0]   # 我们评测是仅支持bs=1
    )

    results_list = []

    for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {args.dataset_name}", disable=(rank != 0))):
        iter_results = {}

        if args.eval_align_depth:
            # 统计align后的scale和shift，作为统计指标
            pred_depth = torch.from_numpy(batch['pred_depth']).to(device_id, non_blocking=True)
            proj_depth = torch.from_numpy(batch['proj_depth']).to(device_id, non_blocking=True)
            aligned_depth, scale, shift = roe_align(pred_depth[None], proj_depth[None], [batch['filename']])
            aligned_depth_cpu = aligned_depth.squeeze(0).cpu().numpy()
            align_stats = DepthDataset._compute_stats(aligned_depth_cpu)

            iter_results.update({
                "scale": scale[0].item(),
                "shift": shift[0].item(),
                "align_depth_max": align_stats["max"],
                "align_depth_min": align_stats["min"],
                "align_depth_mean": align_stats["mean"],
            })

        # 计算 valid_ratio, Abs Rel 和 δ < 1.25
        pred_depth = batch['pred_depth']
        proj_depth = batch['proj_depth']
        gt_depth_mask = (proj_depth > 0) & np.isfinite(proj_depth)
        valid_mask = (pred_depth > 0) & np.isfinite(pred_depth) & gt_depth_mask

        single_frame_metrics = depth_evaluation_numpy(pred_depth, proj_depth, valid_mask)
        if gt_depth_mask.sum() == 0:
            valid_ratio = 0.0
        else:
            valid_ratio = round(valid_mask.sum() / gt_depth_mask.sum() * 100, 2)

        iter_results.update({
            "scene_id": batch['scene_id'],
            "frame_idx": batch['frame_idx'],
            "cam_type": batch['cam_type'],
            "filename": batch['filename'],
            "Abs Rel": single_frame_metrics["Abs Rel"],
            "δ < 1.25": single_frame_metrics["δ < 1.25"],
            "valid_ratio": valid_ratio,

            # 其他统计信息
            "pred_depth_valid_num": batch['pred_depth_valid_num'],
            "proj_depth_valid_num": batch['proj_depth_valid_num'],
            "pred_depth_max": batch['pred_depth_max'],
            "pred_depth_min": batch['pred_depth_min'],
            "pred_depth_mean": batch['pred_depth_mean'],
            "proj_depth_max": batch['proj_depth_max'],
            "proj_depth_min": batch['proj_depth_min'],
            "proj_depth_mean": batch['proj_depth_mean'],
            "proj_depth_uv_std": batch['proj_depth_uv_std'],
            "proj_depth_value_std": batch['proj_depth_value_std'],
        })
        results_list.append(iter_results)

    # 每个 rank 单独保存自己的结果到临时文件
    temp_df = pd.DataFrame(results_list)
    temp_output_path = output_dir / f"eval_info_rank_{rank}.csv"
    temp_df.to_csv(temp_output_path, index=False)
    if rank == 0:
        print(f"\nAll ranks are saving their temporary results...")

    dist.barrier()

    torch.cuda.empty_cache()
    if rank == 0:
        print("All ranks finished. Merging results...")
        all_dfs = []
        for r in range(world_size):
            temp_path = output_dir / f"eval_info_rank_{r}.csv"
            if temp_path.exists():
                try:
                    all_dfs.append(pd.read_csv(temp_path))
                    os.remove(temp_path) # 清理临时文件
                except pd.errors.EmptyDataError:
                    print(f"Warning: Rank {r} result file is empty. Skipping.")
                except Exception as e:
                    print(f"Warning: Could not read or remove {temp_path}. Error: {e}")
            else:
                print(f"Warning: Missing results file from rank {r} (Path: {temp_path})")
        
        if all_dfs:
            final_df = pd.concat(all_dfs)

            initial_count = len(final_df)
            final_df.drop_duplicates(subset=['filename'], inplace=True, keep='first')
            final_count = len(final_df)
            if initial_count > final_count:
                logging.info(f"Removed {initial_count - final_count} duplicate entries based on 'filename'. Final count: {final_count}")

            final_output_path = output_dir / "eval_results_all.csv"
            final_df.to_csv(final_output_path, index=False)
            print(f"Successfully merged {len(final_df)} results into: {final_output_path}")
            
    dist.destroy_process_group()


if __name__ == "__main__":
    main()