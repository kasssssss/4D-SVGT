import torch
import torch.distributed as dist
from torch.utils.data import Sampler, ConcatDataset
from typing import Iterator, List, Tuple
import logging

class GroupedDynamicBatchSampler(Sampler):
    """
    A batch sampler that groups samples by dataset and dynamically determines
    batch parameters (view_num, frame_num, aspect_ratio, batch_size) for each batch.
    """
    def __init__(
        self, 
        dataset: ConcatDataset, 
        # Config params
        max_img_per_gpu: int,
        view_num_range: List[int],     # [min, max]
        aspect_ratio_range: List[float], # [min, max]
        min_frame_num: int = 2,
        # Fixed params for debug
        fixed_batch_size: int = -1,
        fixed_num_frames: int = 0,
        fixed_num_views: int = 0,
        fixed_aspect_ratio: float = 0.0,
        # Standard sampler params
        shuffle: bool = False, 
        seed: int = 42, 
        drop_last: bool = False,
    ):
        
        if not hasattr(dataset, 'cumulative_sizes'):
            raise TypeError("dataset is not a ConcatDataset")

        self.dataset = dataset
        self.max_img_per_gpu = max_img_per_gpu
        self.view_num_range = view_num_range
        self.aspect_ratio_range = aspect_ratio_range
        self.min_frame_num = min_frame_num
        
        # Debug/Fixed Settings
        self.fixed_batch_size = fixed_batch_size
        self.fixed_num_frames = fixed_num_frames
        self.fixed_num_views = fixed_num_views
        self.fixed_aspect_ratio = fixed_aspect_ratio

        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # DDP Info
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        
        # Pre-calculate indices per dataset
        self.indices_by_dataset = []
        for i in range(len(self.dataset.datasets)):
            start_idx = self.dataset.cumulative_sizes[i-1] if i > 0 else 0
            end_idx = self.dataset.cumulative_sizes[i]
            self.indices_by_dataset.append(list(range(start_idx, end_idx)))
        
        self.num_datasets = len(self.indices_by_dataset)
        # Weights for dataset sampling (based on length)
        self.dataset_weights = torch.tensor([len(d) for d in self.indices_by_dataset], dtype=torch.float)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _generate_batch_config(self, sub_dataset, generator: torch.Generator) -> Tuple[int, int, int, float]:
        """
        为一个 Batch 生成随机的 (BatchSize, View, Frame, Aspect)。
        逻辑：先定 View/Frame，再算 BatchSize，保证不爆显存。
        """        
        view_max = sub_dataset.view_num_max
        frame_max = sub_dataset.frame_num_max

        # 使用相同的g，在各个卡上，产生的随机数都是相同的
        view_num = torch.randint(       # 范围是左闭右开
            self.view_num_range[0], 
            min(self.view_num_range[1], view_max) + 1, 
            (1,), generator=generator
        ).item() if self.fixed_num_views <= 0 else self.fixed_num_views
        
        max_f_by_gpu = self.max_img_per_gpu // view_num
        frame_num = torch.randint(
            self.min_frame_num, 
            min(max_f_by_gpu, frame_max) + 1, 
            (1,), generator=generator
        ).item() if self.fixed_num_frames <= 0 else self.fixed_num_frames

        if self.fixed_aspect_ratio > 0:
            aspect_ratio = self.fixed_aspect_ratio
        else:
            low, high = self.aspect_ratio_range
            aspect_ratio_tensor = low + (high - low) * torch.rand(1, generator=generator)
            aspect_ratio = round(aspect_ratio_tensor.item(), 2)                

        batch_size = max(1, self.max_img_per_gpu // (view_num * frame_num)) if self.fixed_batch_size <= 0 else self.fixed_batch_size

        logging.debug("随机到的数据：batch_size: %d, frame_num: %d, view_num: %d, aspect: %.2f", batch_size, frame_num, view_num, aspect_ratio)
        return batch_size, view_num, frame_num, aspect_ratio


    def __iter__(self) -> Iterator[List[Tuple]]:
        # 所有 rank 都使用 g 生成相同的随机排列
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
                
        # shuffle每个数据集内的索引
        processed_indices_by_dataset = []
        for i in range(self.num_datasets):
            indices = self.indices_by_dataset[i]

            if self.shuffle:
                # 则打乱当前数据集的索引，但这里数据量太大了，我们一般不开启shuffle，使用inside random
                processed_indices = [indices[j] for j in torch.randperm(len(indices), generator=g).tolist()]
            else:
                processed_indices = indices

            if self.drop_last:
                total_size = len(processed_indices) // self.num_replicas * self.num_replicas
                processed_indices = processed_indices[:total_size]
            else:
                # 为了保证每个进程拿到的样本数量完全相同，ddp会在数据集的末尾自动重复添加一些样本，使得总样本数能被GPU数量整除。
                # 例如：总样本数1003，GPU数量4。1003无法被4整除。DistributedSampler 会额外从数据集开头复制1个样本，使得总样本数达到1004。
                # 这样，每个GPU都将分到 1004 / 4 = 251 个样本。
                padding_size = self.num_replicas - (len(processed_indices) % self.num_replicas)
                if padding_size != self.num_replicas:
                    processed_indices += processed_indices[:padding_size]

            # 分配给当前 rank 的索引
            rank_indices = processed_indices[self.rank::self.num_replicas]
            processed_indices_by_dataset.append(rank_indices)

        # 状态跟踪，实现每次sample，从各个数据集中按权重随机选择一个数据集，然后在数据集内部采样一个batch
        # 记录每个数据集中，当前 rank 已经采样到哪个位置了
        current_positions = [0] * self.num_datasets
        # 记录每个数据集分配给当前 rank 的总样本数
        rank_lengths = [len(p) for p in processed_indices_by_dataset]

        # --- 开始生成 batch ---
        while True:
            # 找出还有剩余样本的数据集
            available_datasets_indices = [
                i for i, pos in enumerate(current_positions) if pos < rank_lengths[i]
            ]
            
            # 如果所有数据集都采样完了，则结束迭代
            if not available_datasets_indices:
                break

            # 根据剩余数据集的权重进行采样
            available_weights = self.dataset_weights[available_datasets_indices]

            # 使用 torch.multinomial 进行加权随机抽样，所有 rank 使用同一个 generator，保证抽到相同的数据集
            selected_local_idx = torch.multinomial(available_weights, 1, generator=g).item()
            dataset_idx = available_datasets_indices[selected_local_idx]

            sub_dataset = self.dataset.datasets[dataset_idx]
            batch_size, view_num, frame_num, aspect_ratio = self._generate_batch_config(sub_dataset, g)

            # 构建batch
            rank_indices = processed_indices_by_dataset[dataset_idx]
            current_pos = current_positions[dataset_idx]
            end_pos = current_pos + batch_size

            batch_indices = rank_indices[current_pos:end_pos]

            # 如果 batch 为空，这个数据集遍历完了，直接进入下一次循环选择新数据集
            if not batch_indices:
                # 将此数据集标记为完成
                current_positions[dataset_idx] = rank_lengths[dataset_idx]
                continue

            if self.drop_last and len(batch_indices) < batch_size:
                # 如果 drop_last，则这个不完整的 batch 被丢弃，但它的索引已经被消耗
                current_positions[dataset_idx] = end_pos
                continue

            # 更新当前数据集的采样位置
            current_positions[dataset_idx] = end_pos
            
            yield [(idx, view_num, frame_num, aspect_ratio) for idx in batch_indices]
    
    def __len__(self):
        # Return a large dummy length
        return 10000000