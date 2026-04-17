"""Dense dynamic prior branch."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from dvgt_occ.models.decoders import DynamicDenseDecoder
from dvgt_occ.types import DynamicDenseOutput, ReassembledFeatures


class DynamicDenseBranch(nn.Module):
    def __init__(self, channels: int = 256, full_channels: int = 128) -> None:
        super().__init__()
        self.decoder = DynamicDenseDecoder(channels=channels, full_channels=full_channels)
        self.gradient_checkpointing = False
        self.head = nn.Sequential(
            nn.Conv2d(channels + 4, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.full_head = nn.Sequential(
            nn.Conv2d(full_channels, full_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(full_channels, 1, 1),
        )

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, features: ReassembledFeatures, xyz_1_4: torch.Tensor, conf_1_4: torch.Tensor) -> DynamicDenseOutput:
        if self.gradient_checkpointing and self.training:
            h2, p2, full = checkpoint(self.decoder, *features.as_tuple(), use_reentrant=False)
        else:
            h2, p2, full = self.decoder(*features.as_tuple())
        b, t, v, c, h, w = p2.shape
        geom = torch.cat([xyz_1_4, conf_1_4], dim=3)
        x = torch.cat([p2, geom], dim=3).reshape(b * t * v, c + 4, h, w)
        dyn_logit = self.head(x).reshape(b, t, v, 1, h, w)
        full_logit = self.full_head(full.reshape(b * t * v, full.shape[3], full.shape[4], full.shape[5]))
        full_logit = full_logit.reshape(b, t, v, 1, full.shape[4], full.shape[5])
        return DynamicDenseOutput(
            dyn_logit_1_4=dyn_logit,
            dyn_logit_full=full_logit,
            dyn_feat_1_4=p2,
            h2=h2,
            p2=p2,
            full=full,
        )
