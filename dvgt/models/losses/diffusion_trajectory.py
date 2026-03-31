import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import py_sigmoid_focal_loss
from dvgt.utils.trajectory import convert_pose_rdf_to_trajectory_rdf

class DiffusionTrajectoryLoss(nn.Module):

    def __init__(
        self, 
        cls_loss_weight: float = 10.0, 
        reg_loss_weight: float = 8.0,
        anchor_filepath: str = 'data_annotation/anchors/train_total_20/future_anchors_20_rdf.npz',
        gt_scale_factor: float = 0.1,
    ):
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.gt_scale_factor = gt_scale_factor

        anchor_data = np.load(anchor_filepath)
        anchor = torch.from_numpy(anchor_data['centers']).to(torch.float32)     # [N, 8, 2]

        anchor = anchor[None, None].flatten(-2)  # [1, 1, N, 8*2]
        if self.gt_scale_factor != 1.0:
            anchor = anchor * self.gt_scale_factor
        self.register_buffer('anchor', anchor) 

    def forward(self, predictions, batch):
        """
        pred:
            diff_traj_reg_list: List[bs, T, 20, 8, 3]
            diff_traj_cls_list: List[bs, T, 20]
        batch:
            future_ego_n_to_ego_curr: [B, T, 8, 4, 4]
        """
        bs, T, num_mode, future_window, d = predictions['diff_traj_reg_list'][0].shape
        target_traj = convert_pose_rdf_to_trajectory_rdf(batch["future_ego_n_to_ego_curr"])     # [bs, T, 8, 3]

        # NOTE: 'future_ego_n_to_ego_curr' has been downscaled by a factor of 0.1 in the DataLoader.
        # The resulting 'target_traj' is now in the optimal numerical range for direct supervision.
        # Similarly, anchors have been scaled during initialization to maintain consistency.
        dist = torch.linalg.norm(target_traj[..., :2].unsqueeze(2).flatten(-2) - self.anchor, dim=-1)    # [B, T, 20]
        mode_idx = torch.argmin(dist, dim=-1)    # [bs, T]
        cls_target = mode_idx
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, future_window, d)    # [B, T, 1, 8, 2]

        # calculate loss
        loss_dict = {}
        total_loss = 0
        for idx, (traj_reg, traj_cls) in enumerate(zip(predictions['diff_traj_reg_list'], predictions['diff_traj_cls_list'])):
            # Calculate cls loss using focal loss
            target_classes_onehot = torch.zeros_like(traj_cls)
            target_classes_onehot.scatter_(-1, cls_target.unsqueeze(-1), 1)

            # Use py_sigmoid_focal_loss function for focal loss calculation
            cls_loss = py_sigmoid_focal_loss(
                traj_cls,
                target_classes_onehot,
                weight=None,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                avg_factor=None
            )

            # Calculate regression loss
            best_reg = torch.gather(traj_reg, 2, mode_idx).squeeze(2)
            reg_loss = F.l1_loss(best_reg, target_traj)

            loss_dict[f"traj_reg_loss_{idx}"] = reg_loss.item()
            loss_dict[f"traj_cls_loss_{idx}"] = cls_loss.item()
            total_loss += cls_loss * self.cls_loss_weight + reg_loss * self.reg_loss_weight

        return loss_dict, total_loss