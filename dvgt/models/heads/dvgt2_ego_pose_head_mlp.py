# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from dvgt.models.layers import Mlp
from dvgt.models.layers.block import Block
from dvgt.models.heads.head_act import activate_ego_pose
from dvgt.utils.trajectory import convert_rdf_traj_to_flu_traj, convert_pose_rdf_to_trajectory_flu, convert_pose_rdf_to_trajectory_rdf


class DVGT2EgoPoseHeadMLP(nn.Module):
    """
    EgoPoseHead predicts ego to world parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated ego pose tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        max_frames: int = 24,
        future_frame_window: int = 8,
        relative_pose_window: int = 1,
        enable_ego_status: bool = False,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR":
            self.target_pose_dim = 7
        else:
            raise ValueError(f"Unsupported ego pose encoding type: {pose_encoding_type}")
        self.target_traj_dim = 4

        self.hist_encoding = None
        if enable_ego_status:
            self.hist_encoding = nn.Linear(8, dim_in)

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.future_frame_window = future_frame_window
        self.relative_pose_window = relative_pose_window
        self.ego_pose_window = self.future_frame_window + self.relative_pose_window

        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_frames, 1, dim_in))
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

        # Learnable empty ego pose pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.relative_pose_window, self.target_pose_dim))
        self.empty_traj_tokens = nn.Parameter(torch.zeros(1, 1, self.future_frame_window, self.target_traj_dim))
        
        self.embed_pose = nn.Linear(self.target_pose_dim, dim_in)
        self.embed_traj = nn.Linear(self.target_traj_dim, dim_in)
        
        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_pose_dim, drop=0)
        self.traj_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_traj_dim, drop=0)

        cached_time_mask = torch.tril(torch.ones(max_frames, max_frames, dtype=torch.bool))
        self.register_buffer('cached_time_mask', cached_time_mask, persistent=False)

    # The ego pose head is computationally lightweight; KV Cache and streaming logic are unnecessary.
    def forward(
        self, 
        ego_tokens: torch.Tensor,   # (B T V N dim)
        ego_status: torch.Tensor,
        num_iterations: int = 4, 
    ) -> Dict:

        # Concatenate the ego pose tokens of all views in one frame.
        ego_tokens = ego_tokens.mean(2)    # (B T N dim)
        ego_tokens = self.token_norm(ego_tokens)

        T = ego_tokens.shape[1]
        ego_tokens += self.temporal_pos_embed[:, :T]

        return self.trunk_fn(ego_tokens, ego_status, num_iterations)

    def trunk_fn(
        self, 
        ego_tokens: torch.Tensor, 
        ego_status: torch.Tensor, 
        num_iterations: int = 4, 
    ) -> Dict:
        """
        Iteratively refine ego pose pose predictions.
        """
        B, T, N, C = ego_tokens.shape
        ego_tokens = ego_tokens.view(B, T * N, C)
        pred_pose_enc = None
        pred_traj = None

        future_traj_list = []
        relative_pose_enc_list = []

        attn_mask = self.cached_time_mask[:T, :T]
        attn_mask = attn_mask.repeat_interleave(N, dim=0).repeat_interleave(N, dim=1)

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                curr_pose = self.embed_pose(self.empty_pose_tokens.expand(B, T, -1, -1))
                curr_traj = self.embed_traj(self.empty_traj_tokens.expand(B, T, -1, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                curr_pose = self.embed_pose(pred_pose_enc.clone().detach())
                curr_traj = self.embed_traj(pred_traj.clone().detach())
            module_input = torch.cat([curr_pose, curr_traj], dim=2).flatten(1, 2)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            ego_tokens_modulated = gate_msa * modulate(self.adaln_norm(ego_tokens), shift_msa, scale_msa)
            ego_tokens_modulated = ego_tokens_modulated + ego_tokens

            # Attention layers.
            for blk in self.trunk:
                ego_tokens_modulated, _ = blk(ego_tokens_modulated, attn_mask=attn_mask)
            
            ego_tokens_modulated = self.trunk_norm(ego_tokens_modulated).view(B, T, N, C)

            if self.hist_encoding is not None:
                ego_tokens_modulated[:, :, self.relative_pose_window:] += self.hist_encoding(ego_status[:, None])

            # Compute the delta update for the pose encoding and traj.
            pred_pose_enc_delta = self.pose_branch(ego_tokens_modulated[:, :, :self.relative_pose_window])
            pred_traj_delta = self.traj_branch(ego_tokens_modulated[:, :, self.relative_pose_window:])

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
                pred_traj = pred_traj_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
                pred_traj = pred_traj + pred_traj_delta

            # Apply final activation functions for translation, quaternion.
            activated_pose = activate_ego_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act
            )
            future_traj_list.append(pred_traj)      # 这里我们不对theta施加.tanh() * pi的约束，避免复杂的loss，让模型自己去学
            relative_pose_enc_list.append(activated_pose[:, :, 0])

        outputs_dict = {
            "future_traj": future_traj_list[-1],
            "future_traj_list": future_traj_list,
            "relative_ego_pose_enc": relative_pose_enc_list[-1],
            "relative_ego_pose_enc_list": relative_pose_enc_list,
        }
        return outputs_dict

    def post_process_pose_for_eval(
        self,
        predictions: Dict,
    ) -> torch.Tensor:
        return predictions['relative_ego_pose_enc']

    def post_process_traj_for_eval(
        self,
        batch: Dict,
        predictions: Dict,
        outputs: Dict,
    ) -> Tuple[Dict, Dict]:
        pred_traj = predictions['future_traj']
        pred_cos = pred_traj[..., 2:3]
        pred_sin = pred_traj[..., 3:4]
        recovered_theta = torch.atan2(pred_sin, pred_cos) 
        pred_traj = torch.cat([pred_traj[..., :2], recovered_theta], dim=-1)

        outputs['trajectories'] = convert_rdf_traj_to_flu_traj(pred_traj[:, -1])    # [B, 8, 3]
        batch['trajectories'] = convert_pose_rdf_to_trajectory_flu(batch['future_ego_n_to_ego_curr'][:, -1])
        
        # for visualization
        outputs['each_frame_best_traj_rdf'] = pred_traj.clone()
        batch['each_frame_best_traj_rdf'] = convert_pose_rdf_to_trajectory_rdf(batch['future_ego_n_to_ego_curr']).clone()

        if batch['trajectories'].shape[1] != outputs['trajectories'].shape[1]:      # nuscene gt只有6帧
            batch['trajectories'] = F.pad(batch['trajectories'], (0, 0, 0, outputs['trajectories'].shape[1] - batch['trajectories'].shape[1]), 'constant', 0)

        return batch, outputs

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
