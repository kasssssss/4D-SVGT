"""DGGT-style Gaussian head scaffold."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dvgt_occ.models.decoders import GSDenseDecoder, TemporalModule
from dvgt_occ.types import DynamicDenseOutput, GaussianOutput, ReassembledFeatures


class GSHead(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        feature_dim: int = 128,
        instance_dim: int = 32,
        motion_dim: int = 16,
        bias_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.decoder = GSDenseDecoder(channels=channels, full_channels=128)
        self.gradient_checkpointing = False
        out_dim = 3 + 1 + 3 + 4 + 3 + 1 + instance_dim + motion_dim
        self.pre = nn.Conv2d(channels + 3 + 1 + 1 + 3, channels, 1)
        self.pre_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.SiLU(),
        )
        self.temporal = TemporalModule(channels=channels, num_heads=8, num_attention_blocks=2)
        self.post = nn.Sequential(
            nn.Conv2d(channels, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.GELU(), nn.Conv2d(feature_dim, out_dim, 1))
        self.dynamic_head = nn.Conv2d(feature_dim, 1, 1)
        self.instance_dim = instance_dim
        self.motion_dim = motion_dim
        self.bias_scale = float(bias_scale)
        self.feature_dim = int(feature_dim)
        self.offset_limit = 0.35 * self.bias_scale
        self.scale_multiplier = 0.18 * self.bias_scale
        self._init_gaussian_head()

    @staticmethod
    def _inv_softplus(value: float) -> float:
        tensor = torch.tensor(float(value), dtype=torch.float32)
        return float(torch.log(torch.expm1(tensor.clamp_min(1e-6))).item())

    def _init_gaussian_head(self) -> None:
        last = self.head[-1]
        if not isinstance(last, nn.Conv2d):
            return
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            cursor = 0
            cursor += 3  # offset
            last.bias[cursor] = -1.0  # opacity starts visible but below dense-fog levels.
            cursor += 1
            scale_init = max(0.08 / max(self.scale_multiplier, 1e-6), 1e-6)
            last.bias[cursor : cursor + 3].fill_(self._inv_softplus(scale_init))
            cursor += 3
            last.bias[cursor] = 1.0  # identity quaternion in wxyz order.
            cursor += 4
            cursor += 3  # feat_dc residual around the RGB prior.
            last.bias[cursor] = -2.0  # keep_score starts conservative.
        nn.init.zeros_(self.dynamic_head.weight)
        nn.init.constant_(self.dynamic_head.bias, -3.0)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        features: ReassembledFeatures,
        dynamic: Optional[DynamicDenseOutput],
        xyz_1_4: torch.Tensor,
        conf_1_4: torch.Tensor,
        rgb_1_4: Optional[torch.Tensor] = None,
    ) -> GaussianOutput:
        if self.gradient_checkpointing and self.training:
            _, p2, full = checkpoint(self.decoder, *features.as_tuple(), use_reentrant=False)
        else:
            _, p2, full = self.decoder(*features.as_tuple())
        b, t, v, c, h, w = p2.shape
        if dynamic is None:
            dyn = torch.zeros((b, t, v, 1, h, w), device=p2.device, dtype=p2.dtype)
        else:
            dyn = F.interpolate(
                dynamic.dyn_logit_1_4.reshape(b * t * v, 1, *dynamic.dyn_logit_1_4.shape[-2:]),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, t, v, 1, h, w)
        if rgb_1_4 is None:
            rgb_1_4 = torch.zeros((b, t, v, 3, h, w), device=p2.device, dtype=p2.dtype)
        g_in = torch.cat([p2, xyz_1_4, conf_1_4, dyn, rgb_1_4], dim=3)
        g_feat = self.pre(g_in.reshape(b * t * v, c + 8, h, w)).reshape(b, t, v, c, h, w)
        g_feat = self.pre_refine(g_feat.reshape(b * t * v, c, h, w)).reshape(b, t, v, c, h, w)
        g_feat = self.temporal(g_feat)
        g_feat = self.post(g_feat.reshape(b * t * v, c, h, w)).reshape(b, t, v, self.feature_dim, h, w)
        raw = self.head(g_feat.reshape(b * t * v, self.feature_dim, h, w))
        raw = raw.reshape(b, t, v, raw.shape[1], h, w).permute(0, 1, 2, 4, 5, 3).contiguous()
        dynamic_logit = self.dynamic_head(g_feat.reshape(b * t * v, self.feature_dim, h, w))
        dynamic_logit = dynamic_logit.reshape(b, t, v, 1, h, w).permute(0, 1, 2, 4, 5, 3).contiguous()
        cursor = 0
        offset = self.offset_limit * torch.tanh(raw[..., cursor : cursor + 3])
        cursor += 3
        opacity = raw[..., cursor : cursor + 1]
        cursor += 1
        scale_raw = raw[..., cursor : cursor + 3]
        cursor += 3
        rotation = raw[..., cursor : cursor + 4]
        cursor += 4
        feat_dc_raw = raw[..., cursor : cursor + 3]
        cursor += 3
        keep_score = raw[..., cursor : cursor + 1]
        cursor += 1
        instance_affinity = raw[..., cursor : cursor + self.instance_dim]
        cursor += self.instance_dim
        motion_code = raw[..., cursor : cursor + self.motion_dim]
        center = xyz_1_4.permute(0, 1, 2, 4, 5, 3).contiguous() + offset
        scale = self.scale_multiplier * F.softplus(scale_raw)
        if rgb_1_4 is not None:
            rgb_prior = rgb_1_4.permute(0, 1, 2, 4, 5, 3).contiguous().clamp(1e-4, 1.0 - 1e-4)
            feat_dc = torch.sigmoid(feat_dc_raw + torch.logit(rgb_prior))
        else:
            feat_dc = torch.sigmoid(feat_dc_raw)
        return GaussianOutput(
            dense_feat=g_feat.permute(0, 1, 2, 4, 5, 3).contiguous(),
            center=center,
            offset=offset,
            opacity=torch.sigmoid(opacity),
            scale=scale,
            rotation=torch.nn.functional.normalize(rotation, dim=-1),
            feat_dc=feat_dc,
            keep_score=torch.sigmoid(keep_score),
            instance_affinity=instance_affinity,
            motion_code=motion_code,
            aux_decoder_full=full,
            dynamic_logit=dynamic_logit,
        )
