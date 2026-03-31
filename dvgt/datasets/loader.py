from typing import Callable, Optional
from omegaconf import DictConfig, ListConfig
from hydra.utils import instantiate
from pandas.core.computation.ops import Op
from torch.utils.data.dataloader import default_collate
import logging
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from dvgt.datasets.transforms.worker_fn import get_worker_init_fn

class DynamicTorchDataset:
    def __init__(
        self,
        common_dataset_config: DictConfig,
        dataset_config: ListConfig,
        composed_dataset_config: DictConfig,
        num_workers: int,
        training: bool = True,
        sampler_config: Optional[DictConfig] = None,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,   # 这个参数没用，每个新的epoch都会重新创建一个dataloader
        seed: int = 42,
    ) -> None:
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        self.worker_init_fn = worker_init_fn

        # Instantiate the dataset
        logging.info(f"Initializing {len(dataset_config)} sub-datasets (Parquet Loading)...")
        sub_datasets = []
        for ds_conf in dataset_config:
            ds = instantiate(common_dataset_config, **ds_conf)
            sub_datasets.append(ds)

        self.dataset = instantiate(composed_dataset_config, datasets=sub_datasets)

        # Create samplers
        self.batch_sampler = None
        self.sampler = None

        if training and sampler_config is not None:
            self.batch_sampler = instantiate(
                sampler_config,
                dataset=self.dataset.base_dataset,
                seed=seed,
            )
        else:
            if dist.is_available() and dist.is_initialized():
                logging.info("Building DistributedSampler with val dataset")
                self.sampler = DistributedSampler(
                    self.dataset, 
                    shuffle=False,
                    drop_last=False
                )
            else:
                self.sampler = None # Default SequentialSampler

    def get_loader(self, epoch: int):
        logging.info("Building dynamic dataloader with epoch: %d", epoch)

        # Set the epoch for the sampler
        if self.batch_sampler is not None:
            self.batch_sampler.set_epoch(epoch)

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,   # training only
            sampler=self.sampler,               # validation only
            batch_size=1,                       # validation only, currently only support batch_size=1
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )
        