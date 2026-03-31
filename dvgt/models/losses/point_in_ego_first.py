import torch.nn as nn
import torch

from .utils import compute_point_loss

class PointInEgoFirstLoss(nn.Module):

    def __init__(
        self, 
        weight: float = 1.0, 
        reg_loss_type: str = 'l2',
        valid_range: float = 0.98,
        alpha: float = 0.2,
   ):
        super().__init__()
        self.weight = weight
        self.reg_loss_type = reg_loss_type
        self.valid_range = valid_range
        self.alpha = alpha

        self.gradient_loss_fn = 'normal'

    def forward(self, predictions, batch):
        """
        pred:
            points: [B, T, V, H, W, 3]
            points_conf: [B, T, V, H, W]
        batch:
            points_in_ego_first: [B, T, V, H, W, 3]
            point_masks: [B, T, V, H, W]
        """
        point_loss_dict = compute_point_loss(
            pred_points=predictions['points'],
            pred_points_conf=predictions['points_conf'],
            gt_points=batch['points_in_ego_first'],
            gt_points_mask=batch['point_masks'], 
            prefix="ego_first_",
            alpha=self.alpha,
            gradient_loss_fn=self.gradient_loss_fn,
            valid_range=self.valid_range,
            reg_loss_type=self.reg_loss_type,
        )
        total_loss = point_loss_dict["loss_ego_first_conf_point"] + point_loss_dict["loss_ego_first_reg_point"] + point_loss_dict["loss_ego_first_grad_point"]
        total_loss = total_loss * self.weight

        # convert to numpy for logging
        new_point_loss_dict = {}
        for key, value in point_loss_dict.items():
            if isinstance(value, torch.Tensor):
                new_point_loss_dict[key] = value.item()
            else:
                new_point_loss_dict[key] = value
        return point_loss_dict, total_loss