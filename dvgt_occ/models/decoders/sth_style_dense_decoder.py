"""VDA-style STH dense decoder with temporal attention blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, max_frames: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.max_frames = int(max_frames)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, channels))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [N,T,C], got {tuple(x.shape)}")
        _, t, _ = x.shape
        if t > self.max_frames:
            raise ValueError(f"Temporal length {t} exceeds max_frames={self.max_frames}")
        pos = self.pos_embed[:, :t, :].to(dtype=x.dtype, device=x.device)
        h = self.norm1(x + pos)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalModule(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_attention_blocks: int = 2,
        max_frames: int = 32,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TemporalAttentionBlock(
                    channels=channels,
                    num_heads=num_heads,
                    max_frames=max_frames,
                )
                for _ in range(num_attention_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"Expected [B,T,V,C,H,W], got {tuple(x.shape)}")
        b, t, v, c, h, w = x.shape
        y = x.permute(0, 2, 4, 5, 1, 3).reshape(b * v * h * w, t, c)
        for block in self.blocks:
            y = block(y)
        y = y.reshape(b, v, h, w, t, c).permute(0, 4, 1, 5, 2, 3).contiguous()
        return y


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
    def __init__(
        self,
        channels: int = 256,
        full_channels: int = 128,
        output_size: tuple[int, int] = (224, 448),
        num_attention_heads: int = 8,
        num_attention_blocks: int = 2,
        max_frames: int = 32,
    ) -> None:
        super().__init__()
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.half_size = (max(1, self.output_size[0] // 2), max(1, self.output_size[1] // 2))
        self.temporal_f3 = TemporalModule(
            channels,
            num_heads=num_attention_heads,
            num_attention_blocks=num_attention_blocks,
            max_frames=max_frames,
        )
        self.temporal_f4 = TemporalModule(
            channels,
            num_heads=num_attention_heads,
            num_attention_blocks=num_attention_blocks,
            max_frames=max_frames,
        )
        self.temporal_path4 = TemporalModule(
            channels,
            num_heads=num_attention_heads,
            num_attention_blocks=num_attention_blocks,
            max_frames=max_frames,
        )
        self.temporal_path3 = TemporalModule(
            channels,
            num_heads=num_attention_heads,
            num_attention_blocks=num_attention_blocks,
            max_frames=max_frames,
        )
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
        p1 = self.refine1(p2 + f1, size=self.half_size)
        b, t, v, c, h, w = p1.shape
        full = self.full(p1.reshape(b * t * v, c, h, w))
        full = F.interpolate(full, size=self.output_size, mode="bilinear", align_corners=False)
        full = full.reshape(b, t, v, full.shape[1], self.output_size[0], self.output_size[1])
        return h2, p2, full


class DynamicDenseDecoder(STHStyleDenseDecoder):
    pass


class GSDenseDecoder(STHStyleDenseDecoder):
    pass


class OccDenseDecoder(STHStyleDenseDecoder):
    pass
