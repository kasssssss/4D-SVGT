import torch
import torch.nn as nn
import torch.nn.functional as F
from dvgt.utils.trajectory import convert_pose_rdf_to_trajectory_rdf

class TrajectoryLoss(nn.Module):
    """
    TrajectoryLoss is a loss function for trajectory (x, y, cos_theta, sin_theta) prediction.
    """

    def __init__(
        self, 
        xy_weight: float = 1.0, 
        theta_weight: float = 1.0,
        gt_scale_factor: float = 1.0,
        gamma: float = 0.6,  # 引入和 Pose 一致的 temporal decay weight
    ):
        super().__init__()
        self.xy_weight = xy_weight
        self.theta_weight = theta_weight
        self.gt_scale_factor = gt_scale_factor
        self.gamma = gamma

    def forward(self, predictions, batch):
        """
        pred:
            future_traj_list: List[bs, T, 8, 4]
        batch:
            future_ego_n_to_ego_curr: [B, T, 8, 4, 4]
        """
        # 注意，这里的batch["future_ego_n_to_ego_curr"]，已经处理过了scale10，其产生的gt_traj可以直接用于监督
        gt_traj = convert_pose_rdf_to_trajectory_rdf(batch["future_ego_n_to_ego_curr"])     # [bs, T, 8, 3]
        gt_xy = gt_traj[..., :2]
        gt_theta = gt_traj[..., 2]
        gt_sincos = torch.stack([torch.cos(gt_theta), torch.sin(gt_theta)], dim=-1)

        loss_dict = {}
        traj_reg_loss = 0.0
        traj_theta_loss = 0.0
        
        traj_list = predictions['future_traj_list']
        n_stages = len(traj_list)

        for stage_idx, traj in enumerate(traj_list):
            
            # Later stages get higher weight (gamma^0 = 1.0 for final stage)
            stage_weight = self.gamma ** (n_stages - stage_idx - 1)

            pred_xy = traj[..., :2]
            pred_dir = traj[..., 2:4]
            pred_sincos = F.normalize(pred_dir, p=2, dim=-1)

            loss_xy = F.smooth_l1_loss(pred_xy, gt_xy, beta=self.gt_scale_factor, reduction='mean')
            
            # Cosine Similarity Loss
            cos_sim = (pred_sincos * gt_sincos).sum(dim=-1)
            loss_theta = (1.0 - cos_sim).mean()

            # Accumulate weighted losses across stages
            traj_reg_loss += loss_xy * stage_weight
            traj_theta_loss += loss_theta * stage_weight

        # Average over all stages
        traj_reg_loss = traj_reg_loss / n_stages
        traj_theta_loss = traj_theta_loss / n_stages

        loss_dict["traj_reg_loss"] = traj_reg_loss.item()
        loss_dict["traj_theta_loss"] = traj_theta_loss.item()

        total_loss = (traj_reg_loss * self.xy_weight) + (traj_theta_loss * self.theta_weight)

        return loss_dict, total_loss