from pathlib import Path
from typing import Set, Dict

def get_existing_depth_files(storage_image_path: Path, storage_pred_depth_path: Path, storage_proj_depth_path: Path) -> Set[Path]:
    # 排除已经生成proj depth, pred depth和image
    # 需要三个都生成了，才判断为已经生成，否则就重新生成这条数据
    existing_images = storage_image_path.glob(f"**/*.jpg")
    existing_proj_depths = storage_proj_depth_path.glob(f"**/*.png")
    existing_pred_depths = storage_pred_depth_path.glob(f"**/*.png")
    
    relative_images_path = {p.relative_to(storage_image_path).with_suffix('') for p in existing_images}
    relative_proj_depths_path = {p.relative_to(storage_proj_depth_path).with_suffix('') for p in existing_proj_depths}
    relative_pred_depths_path = {p.relative_to(storage_pred_depth_path).with_suffix('') for p in existing_pred_depths}
    
    existing_files = relative_pred_depths_path & relative_images_path & relative_proj_depths_path   # 求交集
    existing_files = {storage_pred_depth_path / p.with_suffix('.png') for p in existing_files}   # 用depth path是否存在作为验证存在文件的标准
    return existing_files

def create_unique_output_dirs(samples_to_process) -> None:
    unique_output_dirs = {
        path
        for s in samples_to_process
        for path in (s.save_pred_depth_path.parent, s.save_image_path.parent, s.save_proj_depth_path.parent)
    }
    for dir_path in unique_output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

def gen_and_create_output_dirs(
    filename: Path, 
    storage_pred_depth_path: Path, 
    storage_proj_depth_path: Path, 
    storage_align_depth_path: Path,
    storage_image_path: Path, 
) -> Dict:
    save_pred_depth_path = storage_pred_depth_path / filename.with_suffix('.png')
    save_proj_depth_path = storage_proj_depth_path / filename.with_suffix('.png')
    save_image_path = storage_image_path / filename.with_suffix('.jpg')
    save_align_depth_path = storage_align_depth_path / filename.with_suffix('.png')
    
    save_pred_depth_path.parent.mkdir(parents=True, exist_ok=True)
    save_proj_depth_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_path.parent.mkdir(parents=True, exist_ok=True)
    save_align_depth_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        'save_pred_depth_path': str(save_pred_depth_path),
        'save_proj_depth_path': str(save_proj_depth_path),
        'save_image_path': str(save_image_path),
        'save_align_depth_path': str(save_align_depth_path),
    }