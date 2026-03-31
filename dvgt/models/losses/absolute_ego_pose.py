import torch.nn as nn
import torch

from .utils import compute_ego_pose_loss

class AbsoluteEgoPoseLoss(nn.Module):

    def __init__(
        self, 
        weight: float = 5.0, 
        T_loss_type: str = 'l1',
        R_loss_type: str = 'l1',
   ):
        super().__init__()
        self.weight = weight
        self.T_loss_type = T_loss_type
        self.R_loss_type = R_loss_type

    def forward(self, predictions, batch):
        """
        不监督第一帧
        pred:
            absolute_ego_pose_enc_list: List[B, T, 7]
        batch:
            ego_past_to_ego_curr: [B, T, 4, 4]
            point_masks: [B, T, V, H, W]
        """
        ego_pose_loss_dict = compute_ego_pose_loss(
            pred_pose_encodings=[pose_enc[:, 1:] for pose_enc in predictions['absolute_ego_pose_enc_list']], 
            point_masks=batch['point_masks'][:, 1:], 
            gt_ego_pose=batch['ego_n_to_ego_first'][:, 1:],
            prefix='absolute_',
            T_loss_type=self.T_loss_type,
            R_loss_type=self.R_loss_type,
        )
        total_loss = ego_pose_loss_dict['loss_absolute_ego_pose_T'] + ego_pose_loss_dict['loss_absolute_ego_pose_R']
        total_loss *= self.weight

        # 转为numpy记录
        new_dict = {}
        for key, value in ego_pose_loss_dict.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = value.item()
            else:
                new_dict[key] = value
        return new_dict, total_loss