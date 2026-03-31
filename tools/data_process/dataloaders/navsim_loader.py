import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from typing import List, Dict

from .factory import register_dataset
from .open_scene_loader import OpenSceneDataset

logger = logging.getLogger(__name__)

def ensure_frame_continuous(scene_id, metas, frame_interval=0.5) -> Dict[str, List]:
    if len(metas) < 2:
        return {scene_id: metas}

    # 检查所有帧是不是都是固定间隔，将不满足时间间隔的片段切分
    timestamps = np.array([m['timestamp'] for m in metas]) / 1e6
    diffs = np.diff(timestamps)
    is_close = np.isclose(diffs, frame_interval, atol=1e-2)

    continue_metas = {}
    if is_close.all():
        continue_metas[scene_id] = metas
    else:
        diff_idxs = np.where(~is_close)[0]
        logging.info(f"{scene_id} is not continuous, split into: {len(diff_idxs)}")

        start = 0
        for i, diff_idx in enumerate(diff_idxs):
            cut_point = diff_idx + 1
            continue_metas[f"{scene_id}#{i}"] = metas[start: cut_point]
            start = cut_point

        continue_metas[f"{scene_id}#{len(diff_idxs)}"] = metas[start: ]
    
    return continue_metas

@register_dataset('navsim')
class NavsimDataset(OpenSceneDataset):

    def __init__(
        self,
        data_root: str = 'public_datasets/openscene',
        storage_pred_depth_path: str = 'public_datasets/data_annotation/pred_depth/moge_v2_large/openscene',
        storage_align_depth_path: str = 'public_datasets/data_annotation/align_depth/moge_v2_large_correct_focal/openscene',
        storage_proj_depth_path: str = 'public_datasets/data_annotation/proj_depth/moge_v2_large_correct_focal/openscene',
        storage_image_path: str = 'public_datasets/data_annotation/image/moge_v2_large_correct_focal/openscene',
        num_tokens: int = 2500,
        split: str = 'test',  # 'train' or 'val' or 'trainval_slice_i' or 'val_slice_i'
        split_quantity: int = 5,    # 训练集太大了，拆分为N份，多机器快速标注
    ) -> None:
        
        super().__init__(
          data_root=data_root, 
          storage_pred_depth_path=storage_pred_depth_path, 
          storage_proj_depth_path=storage_proj_depth_path, 
          storage_image_path=storage_image_path, 
          storage_align_depth_path=storage_align_depth_path, 
          num_tokens=num_tokens,
          split=split,
          split_quantity=split_quantity
        )
        
    def _gather_samples(self):
        if self.split == "test":
            pkl_root_path = self.data_root / "navsim_logs" / self.split
            pkl_paths = list(pkl_root_path.glob("*.pkl"))
        else:
            pkl_root_path = self.data_root / "navsim_logs" / "trainval"
            pkl_paths = list(pkl_root_path.glob("*.pkl"))
            if self.split_quantity > 1:
                size = len(pkl_paths) // self.split_quantity
                x = self.current_split_quantity
                pkl_paths = pkl_paths[(x-1)*size : (x*size if x < self.split_quantity else len(pkl_paths))]

        samples_to_process = []
        for pkl_path in tqdm(pkl_paths, desc="加载meta数据"):
            with open(pkl_path, 'rb') as f:
                metas = pickle.load(f)
            metas = sorted(metas, key=lambda x: x['timestamp'])
            final_mates = ensure_frame_continuous(pkl_path.stem, metas)

            for scene_id, final_meta in final_mates.items():
                frame_idx = 0
                for meta in final_meta:              
                    sub_scene_id = meta['scene_name']
                    if sub_scene_id in self.EXCLUDED_SCENES:
                        # FIXME: 这里存在很多pose异常的的数据，但是navsim的一个log太长了，如果直接删除，会删除过多的数据，后续考虑精细化删除
                        logging.warning(f"{scene_id} has error pose!，len(scene): {len(final_meta)}")
                          
                    self.meta_infos[(scene_id, frame_idx)] = meta
                    
                    for cam_type in self.CAMERA_NAMES:
                        cam_data = meta['cams'][cam_type]
                        filename = Path(cam_data['data_path']).with_suffix('')
                        samples_to_process.append(self.SampleData(
                            scene_id=scene_id,
                            frame_idx=frame_idx,
                            cam_type=cam_type,
                            filename=filename,
                            frame_token=meta['token']
                        ))
                        
                    frame_idx += 1

        logging.info(f"Total samples to process: {len(samples_to_process)}")
        return samples_to_process   

