"""DGGT-style Gaussian head scaffold."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from dvgt_occ.models.decoders import GSDenseDecoder
from dvgt_occ.types import DynamicDenseOutput, GaussianOutput, ReassembledFeatures


class GSHead(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        instance_dim: int = 32,
        motion_dim: int = 16,
        bias_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.decoder = GSDenseDecoder(channels=channels, full_channels=128)
        self.gradient_checkpointing = False
        out_dim = 3 + 1 + 3 + 4 + 3 + 1 + instance_dim + motion_dim
        self.head = nn.Sequential(nn.Conv2d(channels + 1, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, out_dim, 1))
        self.instance_dim = instance_dim
        self.motion_dim = motion_dim
        self.bias_scale = float(bias_scale)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, features: ReassembledFeatures, dynamic: DynamicDenseOutput, xyz_1_8: torch.Tensor) -> GaussianOutput:
        if self.gradient_checkpointing and self.training:
            h2, _, full = checkpoint(self.decoder, *features.as_tuple(), use_reentrant=False)
        else:
            h2, _, full = self.decoder(*features.as_tuple())
        b, t, v, c, h, w = h2.shape
        dyn = torch.nn.functional.interpolate(
            dynamic.dyn_logit_1_4.reshape(b * t * v, 1, *dynamic.dyn_logit_1_4.shape[-2:]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, v, 1, h, w)
        raw = self.head(torch.cat([h2, dyn], dim=3).reshape(b * t * v, c + 1, h, w))
        raw = raw.reshape(b, t, v, raw.shape[1], h, w).permute(0, 1, 2, 4, 5, 3).contiguous()
        cursor = 0
        offset = raw[..., cursor : cursor + 3]
        cursor += 3
        opacity = raw[..., cursor : cursor + 1]
        cursor += 1
        scale_raw = raw[..., cursor : cursor + 3]
        cursor += 3
        rotation = raw[..., cursor : cursor + 4]
        cursor += 4
        feat_dc = raw[..., cursor : cursor + 3]
        cursor += 3
        keep_score = raw[..., cursor : cursor + 1]
        cursor += 1
        instance_affinity = raw[..., cursor : cursor + self.instance_dim]
        cursor += self.instance_dim
        motion_code = raw[..., cursor : cursor + self.motion_dim]
        center = xyz_1_8.permute(0, 1, 2, 4, 5, 3).contiguous() + offset
        scale = torch.exp((scale_raw + self.bias_scale).clamp(min=-6.0, max=4.0))
        return GaussianOutput(
            center=center,
            offset=offset,
            opacity=torch.sigmoid(opacity),
            scale=scale,
            rotation=torch.nn.functional.normalize(rotation, dim=-1),
            feat_dc=torch.sigmoid(feat_dc),
            keep_score=torch.sigmoid(keep_score),
            instance_affinity=instance_affinity,
            motion_code=motion_code,
            aux_decoder_full=full,
        )
