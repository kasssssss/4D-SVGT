import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from numpy import rec
from omegaconf import OmegaConf

import os
from pathlib import Path
from hydra.utils import instantiate
import torch
import pickle
from tqdm import tqdm
from hydra import initialize, compose
from dvgt.utils.general import set_seeds, copy_data_to_device

def main():
    seed_value = 42
    set_seeds(seed_value)

    config = 'eval/dvgt2/geometry_default.yaml'
    infer_window = 10
    dataset_name = 'openscene'
    fixed_num_frames = 40
    future_frame_num = 8
    clip_split_type = 'DVGT1'       # 'DVGT1', 'DVGT2', 'navtest', 'long'
    save_gt = True
    save_pred = True

    with initialize(version_base=None, config_path='../../configs'):
        cfg = compose(config_name=config)

    # 修改配置
    OmegaConf.set_struct(cfg, False)
    cfg.log_dir = os.path.join(cfg.log_dir.replace('eval', 'visual'), dataset_name, f'fix_{fixed_num_frames}_future_{future_frame_num}')
    os.makedirs(cfg.log_dir, exist_ok=True)
    cfg.data_conf.val.common_dataset_config._target_ = 'dvgt.visualization.visual_dataset.VisualParquetDataset'
    cfg.data_conf.val.common_dataset_config.mode = 'visual'
    cfg.data_conf.val.common_dataset_config.fixed_num_frames = fixed_num_frames
    cfg.data_conf.val.common_dataset_config.future_frame_num = future_frame_num
    cfg.data_conf.val.common_dataset_config.enable_depth = True
    cfg.data_conf.val.common_dataset_config.clip_split_type = clip_split_type
    cfg.num_workers = 0

    cfg.model_wrapper_conf.eval_config.eval_traj = True
    cfg.model_wrapper_conf.eval_config.eval_point_and_pose = True
    if infer_window > 0:
        cfg.model_wrapper_conf.eval_config.infer_window = infer_window
    
    cfg.data_conf.val.dataset_config = [
        d_conf for d_conf in cfg.data_conf.val.dataset_config if dataset_name in d_conf['parquet_path']
    ]

    # ------ 使用train.parquet --------
    # cfg.data_conf.val.dataset_config = [
    #     d_conf for d_conf in cfg.data_conf.train.dataset_config if dataset_name in d_conf['parquet_path']
    # ]
    # cfg.data_conf.val.dataset_config[0].fixed_aspect_ratio = 1.77
    # cfg.data_conf.val.dataset_config[0].dataset_len = -1
    # cfg.data_conf.val.dataset_config[0].frame_num_max = 999
    # ---------------------------------

    device = torch.device('cuda')

    if save_pred:
        # Load Model
        model_wrapper = instantiate(cfg.model_wrapper_conf, _recursive_=False)
        model_wrapper.load(cfg.checkpoint_path)
        model_wrapper.model.to(device).eval()

    # dataset
    val_dataset = instantiate(cfg.data_conf.get('val', None), _recursive_=False)
    val_dataset.seed = seed_value

    dataloader = val_dataset.get_loader(epoch=0)

    for batch in tqdm(dataloader):
        visual_base_dir = Path(cfg.log_dir) / batch['seq_name'][0]
        visual_base_dir.mkdir(parents=True, exist_ok=True)

        batch = copy_data_to_device(batch, device, non_blocking=True)

        save_data_dict = {}

        if save_pred:
            with torch.no_grad():
                batch, predictions = model_wrapper.infer(batch)

            save_data_dict['pred_points'] = predictions['points_in_ego_first'].cpu().numpy()
            save_data_dict['pred_points_conf'] = predictions['points_in_ego_first_conf'].cpu().numpy()
            save_data_dict['pred_ego_n_to_ego_first'] = predictions['ego_n_to_ego_first'].cpu().numpy()
            save_data_dict['pred_ray_depth'] = predictions['ray_depth'].cpu().numpy()

            if "top3_traj" in predictions:
                save_data_dict['top3_traj'] = predictions['top3_traj'].cpu().numpy()
                save_data_dict['top3_traj_idxs'] = predictions['top3_traj_idxs'].cpu().numpy()
            
            if "trajectories" in predictions:
                save_data_dict['pred_traj'] = predictions['trajectories'].cpu().numpy()
            
            if "each_frame_best_traj_rdf" in predictions:
                save_data_dict['pred_each_frame_best_traj_rdf'] = predictions['each_frame_best_traj_rdf'].cpu().numpy()

        if save_gt:
            save_data_dict['points'] = batch['points_in_ego_first'].cpu().numpy()
            save_data_dict['ego_n_to_ego_first'] = batch['ego_n_to_ego_first'].cpu().numpy()  
            save_data_dict['ray_depth'] = batch['ray_depth'].cpu().numpy()
            save_data_dict['intrinsics'] = batch['intrinsics'].cpu().numpy()
            save_data_dict['ego_first_to_cam_n'] = batch['ego_first_to_cam_n'].cpu().numpy()
            
            if "trajectories" in batch:
                save_data_dict['gt_traj'] = batch['trajectories'].cpu().numpy()

            if "each_frame_best_traj_rdf" in batch:
                save_data_dict['gt_each_frame_best_traj_rdf'] = batch['each_frame_best_traj_rdf'].cpu().numpy()

        save_data_dict['cam_types'] = batch['cam_types']
        save_data_dict['images'] = batch['images'].cpu().numpy()
        save_data_dict['depths'] = batch['depths'].cpu().numpy()
        save_data_dict['seq_name'] = batch['seq_name']
        save_data_dict['point_masks'] = batch['point_masks'].cpu().numpy()
        save_path = visual_base_dir / 'saved_data.pkl'

        print(f"正在将数据字典保存到: {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(save_data_dict, f)

if __name__ == "__main__":
    main()