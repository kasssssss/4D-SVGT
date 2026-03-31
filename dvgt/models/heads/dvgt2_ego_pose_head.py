# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dvgt.models.heads.trajectory_pose_decoder import TrajectoryPoseDecoder
from dvgt.models.layers.block import Block
from dvgt.models.layers import Mlp
from dvgt.utils.trajectory import convert_rdf_traj_to_flu_traj, convert_pose_rdf_to_trajectory_flu

class DVGT2EgoPoseHead(nn.Module):

    def __init__(
        self,
        dim_in: int = 3096,
        diffusion_dim_in: int = 256,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        traj_anchor_filepath: str = 'data_annotation/anchors/train_total_20/future_anchors_20_rdf.npz',
        pose_translation_anchor_filepath: str = 'data_annotation/anchors/train_total_20/past_anchors_20_rdf.npz',
        future_frame_window: int = 8,
        relative_pose_window: int = 1,
        max_frames: int = 48,
        enable_ego_status: bool = False,
        gt_scale_factor: float = 0.1,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR":
            self.target_dim = 7
        else:
            raise ValueError(f"Unsupported ego pose encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.future_frame_window = future_frame_window
        self.relative_pose_window = relative_pose_window
        self.ego_pose_window = self.future_frame_window + self.relative_pose_window

        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_frames, 1, 1, dim_in))
        nn.init.normal_(self.temporal_pos_embed, std=1e-6)

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.ModuleList(
            [
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for ego pose token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=diffusion_dim_in, drop=0)

        cached_time_mask = torch.tril(torch.ones(max_frames, max_frames, dtype=torch.bool))
        self.register_buffer('cached_time_mask', cached_time_mask, persistent=False)

        self.traj_pose_decoder = TrajectoryPoseDecoder(
            dim_in=diffusion_dim_in,
            traj_anchor_filepath=traj_anchor_filepath,
            pose_translation_anchor_filepath=pose_translation_anchor_filepath,
            future_frame_window=future_frame_window,
            enable_ego_status=enable_ego_status,
            gt_scale_factor=gt_scale_factor
        )

    # The ego pose head is computationally lightweight; KV Cache and streaming logic are unnecessary.
    def forward(
        self, 
        ego_tokens: torch.Tensor,
        ego_status: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Temporal positional embedding.
        T = ego_tokens.shape[1]
        ego_tokens += self.temporal_pos_embed[:, :T]

        ego_tokens = self.token_norm(ego_tokens)

        return self.trunk_fn(ego_tokens, ego_status)

    def trunk_fn(
        self, 
        pose_tokens: torch.Tensor, 
        ego_status: torch.Tensor,
    ) -> Dict:
        """
        self attention ego pose pose predictions.
        """
        B, T, V, P, C = pose_tokens.shape
        pose_tokens = pose_tokens.view(B, T * V * P, C)

        attn_mask = self.cached_time_mask[:T, :T]
        attn_mask = attn_mask.repeat_interleave(V * P, dim=0).repeat_interleave(V * P, dim=1)

        # Attention layers.
        for blk in self.trunk:
            pose_tokens, _ = blk(pose_tokens, attn_mask=attn_mask)
        
        pose_tokens = self.pose_branch(self.trunk_norm(pose_tokens))

        pose_tokens = pose_tokens.view(B, T, V, P, -1)

        return self.traj_pose_decoder(pose_tokens, ego_status)

    def post_process_pose_for_eval(
        self,
        predictions: Dict,
    ) -> torch.Tensor:
        pose_scores = predictions['diff_poses_cls_list'][-1]    # [B, T, 20]
        pose = predictions['diff_poses_reg_list'][-1]           # [B, T, 20, 7]
        best_mode_idx = pose_scores.argmax(dim=-1)
        best_mode_idx_pose = best_mode_idx[..., None, None].repeat(1, 1, 1, pose.shape[-1])
        best_pose = torch.gather(pose, 2, best_mode_idx_pose).squeeze(2)    # [B, T, 7]
        return best_pose

    def post_process_traj_for_eval(
        self,
        batch: Dict,
        predictions: Dict,
        outputs: Dict,
    ) -> Tuple[Dict, Dict]:

        traj_scores = predictions['diff_traj_cls_list'][-1]     # [B, T, 20]
        traj = predictions['diff_traj_reg_list'][-1]            # [B, T, 20, 8, 3]
        _, top3_traj_idxs = torch.topk(traj_scores, 3)
        top3_traj_idxs_expanded = top3_traj_idxs[..., None, None].expand(-1, -1, -1, self.future_frame_window, 3)
        top3_traj = torch.gather(traj, 2, top3_traj_idxs_expanded)  # [B, T, 3, 8, 3]
        outputs['top3_traj'] = convert_rdf_traj_to_flu_traj(top3_traj[:, -1])     # [B, 3, 8, 3]
        outputs['top3_traj_idxs'] = top3_traj_idxs[:, -1]  # [B, 3, 1, 1]
        
        outputs['trajectories'] = outputs['top3_traj'][:, 0].clone()    # [B, 8, 3]
        batch['trajectories'] = convert_pose_rdf_to_trajectory_flu(batch['future_ego_n_to_ego_curr'][:, -1])
        if batch['trajectories'].shape[1] != outputs['trajectories'].shape[1]:      # The official nuScenes planning metric evaluates only a 6-frame (3s) horizon
            batch['trajectories'] = F.pad(batch['trajectories'], (0, 0, 0, self.future_frame_window - batch['trajectories'].shape[1]), 'constant', 0)

        return batch, outputs