import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch.multiprocessing as mp

import argparse
from hydra import initialize, compose
from dvgt.engine.evaluator import Evaluator

def main():
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument("--config", type=str, default="eval/dvgt/default.yaml", help="Name of the config file")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    # 允许通过命令行覆盖参数，例如: model.depth=12 data.batch_size=4
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options from command line")

    args = parser.parse_args()

    with initialize(version_base=None, config_path='../configs'):
        overrides = args.opts if args.opts else []
        cfg = compose(config_name=args.config, overrides=overrides)

    if args.debug:
        cfg.log_dir += "_debug"
        import shutil   # 清空debug日志
        shutil.rmtree(cfg.log_dir, ignore_errors=True)
        cfg.logging_conf.log_level_primary = "DEBUG"
        
        cfg.data_conf.val.dataset_config = [cfg.data_conf.val.dataset_config[0]]

        cfg.num_workers = 0

        # cfg.checkpoint_path = ''
        # cfg.model_wrapper_conf.model_config.dino_v3_weight_path = None

        # 设置clip=16，方便测试时间
        # cfg.data_conf.val.common_dataset_config.fixed_num_frames = 16
        # cfg.data_conf.val.common_dataset_config.clip_split_type = 'DVGT-2'
        # cfg.data_conf.val.common_dataset_config.future_frame_num = 0

    evaluator = Evaluator(**cfg)
    evaluator.run()


if __name__ == "__main__":
    main()