import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import py_sigmoid_focal_loss
from dvgt.utils.pose_encoding import encode_pose

class DiffusionPastPoseLoss(nn.Module):

    def __init__(
        self, 
        cls_loss_weight: float = 10.0, 
        reg_loss_weight: float = 8.0,
        anchor_filepath: str = 'data_annotation/anchors/train_total_20/past_anchors_20_rdf.npz',
        gt_scale_factor: float = 0.1,
    ):
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.gt_scale_factor = gt_scale_factor
        
        anchor_data = np.load(anchor_filepath)
        anchor = torch.from_numpy(anchor_data['centers']).to(torch.float32)     # [N, 3]

        anchor = anchor[None, None]  # [1, 1, N, 3]
        if self.gt_scale_factor != 1.0:
            anchor = anchor * self.gt_scale_factor
        self.register_buffer('anchor', anchor)

    def forward(self, predictions, batch, pose_encoding_type="absT_quaR"):
        """
        Excluding the first frame from the loss
        pred:
            diff_poses_reg_list: List[bs, T, 20, 7]
            diff_poses_cls_list: List[bs, T, 20]
        batch:
            ego_past_to_ego_curr: [B, T, 4, 4]
        """
        # Excluding the first frame from the loss
        diff_poses_reg_list = [pose[:, 1:] for pose in predictions['diff_poses_reg_list']]
        diff_poses_cls_list = [pose_cls[:, 1:] for pose_cls in predictions['diff_poses_cls_list']]

        # NOTE: 'ego_past_to_ego_curr' has been downscaled by a factor of 0.1 in the DataLoader.
        # The resulting 'target_traj' is now in the optimal numerical range for direct supervision.
        # Similarly, anchors have been scaled during initialization to maintain consistency.
        gt_pose = batch["ego_past_to_ego_curr"][:, 1:]
        gt_pose = encode_pose(gt_pose, pose_encoding_type=pose_encoding_type)   # [B, T, 7]
    
        B, T, N, D = diff_poses_reg_list[0].shape
        dist = torch.linalg.norm(gt_pose[..., :3].unsqueeze(2) - self.anchor, dim=-1)    # [B, T, 20]
        mode_idx = torch.argmin(dist, dim=-1)    # [bs, T]
        cls_target = mode_idx
        mode_idx = mode_idx[..., None, None].repeat(1, 1, 1, D)    # [B, T, 1, 7]

        # calculate loss
        loss_dict = {}
        total_loss = 0
        for idx, (pose_reg, pose_cls) in enumerate(zip(diff_poses_reg_list, diff_poses_cls_list)):
            # Calculate cls loss using focal loss
            target_classes_onehot = torch.zeros_like(pose_cls)
            target_classes_onehot.scatter_(-1, cls_target.unsqueeze(-1), 1)

            # Use py_sigmoid_focal_loss function for focal loss calculation
            cls_loss = py_sigmoid_focal_loss(
                pose_cls,
                target_classes_onehot,
                weight=None,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                avg_factor=None
            )

            # Calculate regression loss
            best_reg = torch.gather(pose_reg, 2, mode_idx).squeeze(2)
            reg_loss = F.l1_loss(best_reg, gt_pose)

            loss_dict[f"relative_pose_reg_loss_{idx}"] = reg_loss.item()
            loss_dict[f"relative_pose_cls_loss_{idx}"] = cls_loss.item()
            total_loss += cls_loss * self.cls_loss_weight + reg_loss * self.reg_loss_weight

        return loss_dict, total_loss