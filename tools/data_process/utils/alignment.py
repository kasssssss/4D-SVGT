# import sys
# sys.path.append("/mnt/ad-alg/planning-users/xiezixun/vggt_work_dir/MoGe")
# sys.path.append("/mnt/ad-alg/planning-users/xiezixun/vggt_work_dir/dvgt")
# sys.path.append("/mnt/ad-alg/planning-users/xiezixun/vggt_work_dir/dvgt/data_process/utils")

from third_party.MoGe.moge.utils.alignment import align_depth_affine
from tools.data_process.utils.image import sample_sparse_depth

from typing import List, Tuple
import torch
import logging
logging = logging.getLogger(__name__)

def roe_align(
    preds: torch.Tensor, targets: torch.Tensor, save_align_depth_path: List[str], sample_count: int = 25000
) -> Tuple[torch.Tensor, List, List]:
    """
    使用 MoGe中的 ROE 对预测值和目标值进行尺度和位移对齐。
    这里只能bs=1处理，大了会爆显存
    preds: [bs, h, w]
    targets: [bs, h, w]，target不合理的值需要是inf
    sample_count: 这里如果lidar过多，ROE的时候也会爆显存，需要随机采样其中一些点来align
    """
    scales = []
    shifts = []
    aligned_depths = []
    device = preds.device

    for pred, target, save_image_path in zip(preds, targets, save_align_depth_path):
        if target.isfinite().sum() > sample_count:
            # 这里如果lidar过多，随机采样一部分用来align
            target = sample_sparse_depth(target, sample_count=sample_count)

        valid_mask = (target > 0) & (pred > 0) & \
                    torch.isfinite(pred) & torch.isfinite(target)

        if valid_mask.sum() >= 2:
            # 至少需要2个点，才能align
            valid_pred = pred[valid_mask]
            valid_target = target[valid_mask]
            scale, shift = align_depth_affine(valid_pred, valid_target, 1 / valid_target)
            aligned_depth = pred * scale + shift
        else:
            # 替换为lidar真值
            scale = shift = torch.tensor(-1, device=device)
            logging.error(f"valid_mask.sum() < 2, pred_mask={pred.isfinite().sum()}, target_mask={target.isfinite().sum()}, save_image_path={save_image_path}")
            aligned_depth = target

        scales.append(scale)
        shifts.append(shift)
        aligned_depths.append(aligned_depth)

    # 避免“下冲现象”
    aligned_depths = torch.stack(aligned_depths).clamp(0)
    
    return aligned_depths, scales, shifts

if  __name__ == "__main__":
    from tools.data_process.utils.io import read_depth
    proj_p = 'public_datasets/data_annotation/proj_depth/moge_v2_large_correct_focal/ddad/000000/rgb/CAMERA_05/15616458249936530.png'
    proj_depth, _ = read_depth(proj_p)
    proj_depth = torch.from_numpy(proj_depth).unsqueeze(0).to('cuda')
    pred_p = 'public_datasets/data_annotation/pred_depth/moge_v2_large/ddad/000000/rgb/CAMERA_05/15616458249936530.png'
    pred_depth, _ = read_depth(pred_p)
    pred_depth = torch.from_numpy(pred_depth).unsqueeze(0).to('cuda')
    align_p = 'public_datasets/data_annotation/align_depth/moge_v2_large_correct_focal/ddad/000000/rgb/CAMERA_05/15616458249936530.png'
    align_depth, _ = read_depth(align_p)
    align_depth = torch.from_numpy(align_depth).unsqueeze(0).to('cuda')

    aligned_depths, scales, shifts = roe_align(
        pred_depth, proj_depth, ['hh']
    )
    print('hh')

