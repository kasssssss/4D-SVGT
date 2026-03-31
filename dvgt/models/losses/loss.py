# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from hydra.utils import instantiate
from typing import Any, Dict, Tuple

class MultitaskLoss(nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    """
    def __init__(self, loss_config: Dict[str, Any]):
        super().__init__()
        self.losses = nn.ModuleList()
        for config in loss_config:
            self.losses.append(instantiate(config))

    def forward(self, predictions, batch) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0.0
        loss_dict = {}

        if self.losses is not None:
            for loss_module in self.losses:
                sub_dict, loss_scalar = loss_module(predictions, batch)
                total_loss += loss_scalar
                loss_dict.update(sub_dict)

        loss_dict["loss_objective"] = total_loss.item()
        return loss_dict, total_loss