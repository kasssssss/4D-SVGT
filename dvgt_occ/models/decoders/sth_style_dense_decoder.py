"""VDA-inspired STH-style dense decoder template.

The temporal block is deliberately lightweight in this scaffold. It keeps the
shape contract and zero-init residual behavior while leaving room to swap in
the full Video Depth Anything implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"Expected [B,T,V,C,H,W], got {tuple(x.shape)}")
        b, t, v, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1, 4, 5).reshape(b * v, c, t, h, w)
        y = self.pointwise(self.depthwise(y))
        y = y.reshape(b, v, c, t, h, w).permute(0, 3, 1, 2, 4, 5).contiguous()
        return x + y


class RefineBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        b, t, v, c, h, w = x.shape
        y = x.reshape(b * t * v, c, h, w)
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        y = self.net(y)
        return y.reshape(b, t, v, c, size[0], size[1])


class STHStyleDenseDecoder(nn.Module):
    def __init__(self, channels: int = 256, full_channels: int = 128) -> None:
        super().__init__()
        self.temporal_f3 = TemporalModule(channels)
        self.temporal_f4 = TemporalModule(channels)
        self.temporal_path4 = TemporalModule(channels)
        self.temporal_path3 = TemporalModule(channels)
        self.refine4 = RefineBlock(channels)
        self.refine3 = RefineBlock(channels)
        self.refine2 = RefineBlock(channels)
        self.refine1 = RefineBlock(channels)
        self.full = nn.Conv2d(channels, full_channels, 3, padding=1)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor):
        h3_pre = self.temporal_f3(f3)
        h4_pre = self.temporal_f4(f4)
        p4 = self.refine4(h4_pre, size=h3_pre.shape[-2:])
        h3 = self.temporal_path4(p4)
        p3 = self.refine3(h3 + h3_pre, size=f2.shape[-2:])
        h2 = self.temporal_path3(p3)
        p2 = self.refine2(h2 + f2, size=f1.shape[-2:])
        p1 = self.refine1(p2 + f1, size=(112, 224))
        b, t, v, c, h, w = p1.shape
        full = self.full(p1.reshape(b * t * v, c, h, w))
        full = F.interpolate(full, size=(224, 448), mode="bilinear", align_corners=False)
        full = full.reshape(b, t, v, full.shape[1], 224, 448)
        return h2, p2, full


class DynamicDenseDecoder(STHStyleDenseDecoder):
    pass


class GSDenseDecoder(STHStyleDenseDecoder):
    pass


class OccDenseDecoder(STHStyleDenseDecoder):
    pass
