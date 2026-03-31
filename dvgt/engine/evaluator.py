import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict
import logging
import datetime
from omegaconf import DictConfig
from hydra.utils import instantiate

from dvgt.evaluation.utils.metric_utils import summarize_and_log, compute_all_metrics
from dvgt.utils.logging import setup_logging
from dvgt.utils.general import safe_makedirs, set_seeds, copy_data_to_device

class Evaluator:
    def __init__(
        self, 
        *,
        data_conf: DictConfig,
        logging_conf: DictConfig,
        model_wrapper_conf: DictConfig,
        enable_trajectory_save: bool = True,
        checkpoint_path: str = '',
        seed_value: int = 42,
        **kwargs,
    ):
        self.logging_conf = logging_conf
        self.model_wrapper_conf = model_wrapper_conf
        self.data_conf = data_conf
        self.seed_value = seed_value
        self.enable_trajectory_save = enable_trajectory_save

        self._setup_distributed()
        set_seeds(self.seed_value)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )

        # Load Model
        logging.info(f"Loading model wrapper: {self.model_wrapper_conf._target_}")
        self.model_wrapper = instantiate(self.model_wrapper_conf, _recursive_=False)
        self.model_wrapper.load(checkpoint_path)
        self.model_wrapper.model.to(self.device).eval()

        # dataset
        self.val_dataset = instantiate(self.data_conf.get('val', None), _recursive_=False)
        self.val_dataset.seed = self.seed_value

    def _setup_distributed(self):  
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.device)
        dist.init_process_group(
            backend="nccl", 
            timeout=datetime.timedelta(minutes=60), 
            rank=self.rank, 
            world_size=self.world_size,
        )
        
    def run(self):
        dataloader = self.val_dataset.get_loader(epoch=0)
        all_metrics = defaultdict(list)

        logging.info("Starting inference loop...")
        progress_bar = tqdm(dataloader, disable=(self.rank != 0), desc="Evaluating")


        for i, batch in enumerate(progress_bar):
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            with torch.no_grad():
                gts, predictions = self.model_wrapper.infer(batch)
                
                scene_metrics = compute_all_metrics(gts, predictions, enable_trajectory_save=self.enable_trajectory_save)

                for k, v in scene_metrics.items():
                    all_metrics[k].extend(v)

        dist.barrier()
        summarize_and_log(
            rank=self.rank,
            world_size=self.world_size,
            log_dir=self.logging_conf.log_dir,
            all_metrics=all_metrics,
        )