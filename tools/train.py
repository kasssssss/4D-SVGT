import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from hydra import initialize, compose
from dvgt.engine.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument("--config", type=str, default="train/dvgt/default.yaml", help="Name of the config file")
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
        cfg.logging_conf.log_freq = 1
        cfg.logging_conf.log_visual_frequency.train = 5
        cfg.logging_conf.log_visual_frequency.val = 5

        cfg.data_conf.train.dataset_config = [cfg.data_conf.train.dataset_config[-1]]
        cfg.data_conf.val.dataset_config = [cfg.data_conf.val.dataset_config[-1]]

        cfg.limit_train_batches = 5
        cfg.limit_val_batches = 5
        cfg.val_epoch_freq = 1

        cfg.checkpoint.save_freq = 1
        # cfg.checkpoint.direct_load_pretrained_weights_path = None
        # cfg.model.dino_v3_weight_path = None

        cfg.num_workers = 0

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()