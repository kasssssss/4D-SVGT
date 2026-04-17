"""Auxiliary global latent bridge between Gaussian and Occ features."""

import torch
import torch.nn as nn


class GSOccGlobalLatentBridge(nn.Module):
    def __init__(self, channels: int = 256, num_latents: int = 128) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, channels) * 0.02)
        self.cross = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)

    def forward(self, gs_tokens: torch.Tensor, occ_tokens: torch.Tensor) -> torch.Tensor:
        b = gs_tokens.shape[0]
        latent = self.latents.unsqueeze(0).expand(b, -1, -1)
        context = torch.cat([gs_tokens, occ_tokens], dim=1)
        updated, _ = self.cross(latent, context, context)
        return updated
