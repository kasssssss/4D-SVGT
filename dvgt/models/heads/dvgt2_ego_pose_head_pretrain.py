# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional
import torch
import torch.nn as nn

from dvgt.models.layers import Mlp
from dvgt.models.layers.block import Block
from dvgt.models.heads.head_act import activate_ego_pose

class DVGT2EgoPoseHeadPretrain(nn.Module):
    
    def __init__(
        self,
        dim_in: int = 3096,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        ego_pose_window: int = 1,
        max_frames: int = 48,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR":
            self.target_dim = 7
        else:
            raise ValueError(f"Unsupported ego pose encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.trunk_depth = trunk_depth
        self.ego_pose_window = ego_pose_window

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

        # Adaptive layer normalization without affine parameters.
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    # The ego pose head is computationally lightweight; KV Cache and streaming logic are unnecessary.
    def forward(
        self, 
        ego_tokens: torch.Tensor,   # (B T V P dim)
        ego_status: Optional[torch.Tensor] = None       # Not used in this model.
    ) -> Dict:
        # Concatenate the ego pose tokens of all views in one frame.
        T = ego_tokens.shape[1]
        ego_tokens += self.temporal_pos_embed[:, :T]

        ego_tokens = self.token_norm(ego_tokens)

        return self.trunk_fn(ego_tokens)

    def trunk_fn(self, pose_tokens: torch.Tensor) -> Dict:
        B, T, V, P, C = pose_tokens.shape
        pose_tokens = pose_tokens.view(B, T * V * P, C)

        for blk in self.trunk:
            pose_tokens, _ = blk(pose_tokens)

        pose_tokens = pose_tokens.view(B, T, V * P, C)
        pose_tokens = pose_tokens.mean(dim=2)
        
        pose_tokens = self.pose_branch(self.trunk_norm(pose_tokens))

        # Apply final activation functions for translation, quaternion.
        activated_pose = activate_ego_pose(
            pose_tokens, trans_act=self.trans_act, quat_act=self.quat_act
        )

        # Align tensor shapes for loss calculation.
        output_dict = {
            "relative_ego_pose_enc": activated_pose,
            "relative_ego_pose_enc_list": [activated_pose]
        }

        return output_dict

    def post_process_pose_for_eval(
        self,
        predictions: Dict,
    ) -> torch.Tensor:
        return predictions['relative_ego_pose_enc']
        