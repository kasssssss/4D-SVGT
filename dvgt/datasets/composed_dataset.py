import torch
import random
import bisect
from typing import List, Tuple, Dict, Optional, Union
from torch.utils.data import Dataset, ConcatDataset
from dvgt.datasets.transforms.augmentation import get_image_augmentation
from einops import rearrange

class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple. 
    负责路由，根据global_idx，解析出local_idx，

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (global_idx, view_num, frame_num, aspect),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.
    """

    def __getitem__(self, idx: Union[int, Tuple]):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 4 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # global index

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if idx_tuple:
            new_tuple = (sample_idx,) + idx_tuple[1:]
            return self.datasets[dataset_idx][new_tuple]
        else:
            return self.datasets[dataset_idx][sample_idx]

class ComposedDataset(Dataset):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset provides a flexible way to combine multiple base datasets while
    applying shared augmentations, track generation, and other processing steps.
    It handles image normalization, tensor conversion, and other preparations
    needed for training computer vision models with sequences of images.
    """
    def __init__(
        self, 
        datasets: List[Dataset],
        training: bool = True,
        # Augmentation Params
        cojitter: bool = False,
        cojitter_ratio: float = 0.0,
        color_jitter: Optional[Dict[str, float]] = None,
        gray_scale: bool = False,
        gau_blur: bool = False,
    ):
        """
        Args:
            datasets: 已经实例化好的数据集列表
            training: 是否训练模式
            cojitter: 是否对序列所有帧应用相同的 Jitter
            cojitter_ratio: Co-Jitter 的触发概率
            color_jitter: 颜色抖动参数字典 (brightness, contrast, etc.)
            gray_scale: 是否启用随机灰度
            gau_blur: 是否启用高斯模糊
        """
        self.base_dataset = TupleConcatDataset(datasets)
        self.training = training
        self.total_samples = len(self.base_dataset)

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.cojitter = cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = cojitter_ratio
        
        self.image_aug = None
        if training:
            # Initialize image augmentations (color jitter, grayscale, gaussian blur)
            self.image_aug = get_image_augmentation(
                color_jitter=color_jitter,
                gray_scale=gray_scale,
                gau_blur=gau_blur,
            )

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples


    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors, applies augmentations,
        and prepares tracks if enabled.

        Args:
            idx_tuple (tuple): a tuple of (seq_index, views_per_frame, aspect_ratio, frame_per_seq)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, tracks, etc.).
        """
        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        
        # Normalize images from [0, 255] to [0, 1]
        images = torch.from_numpy(batch['images'])
        images = rearrange(images, 't v h w c -> t v c h w')
        images = images.float() / 255.0

        T, V = images.shape[:2]

        # --- Apply Color Augmentation (training mode only) ---
        if self.training and self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                # Apply the same color jittering transformation to all frames
                images = self.image_aug(images)
            else:
                # Apply different color jittering to each view individually
                for t in range(T):
                    for v in range(V):
                        images[t, v] = self.image_aug(images[t, v])

        batch['images'] = images

        return batch