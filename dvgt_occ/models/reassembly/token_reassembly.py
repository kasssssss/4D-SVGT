"""Shared token reassembly from frozen DVGT tokens to four 2D feature scales."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dvgt_occ.types import ReassembledFeatures


class TokenReassembly(nn.Module):
    def __init__(
        self,
        selected_layers: Sequence[int] = (4, 11, 17, 23),
        in_dim: int = 4096,
        out_dim: int = 256,
        patch_grid: Sequence[int] = (14, 28),
        special_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.selected_layers = tuple(selected_layers)
        self.patch_grid = tuple(patch_grid)
        self.special_tokens = special_tokens
        self.projections = nn.ModuleDict({str(layer): nn.Conv2d(in_dim, out_dim, 1) for layer in self.selected_layers})

    def forward(
        self,
        aggregated_tokens: Mapping[int, torch.Tensor],
        raw_patch_tokens: torch.Tensor,
    ) -> ReassembledFeatures:
        projected = []
        for layer in self.selected_layers:
            if layer not in aggregated_tokens:
                raise KeyError(f"Missing DVGT layer {layer}; available={list(aggregated_tokens.keys())}")
            tokens = aggregated_tokens[layer]
            projected.append(self._project(tokens, raw_patch_tokens, self.projections[str(layer)]))

        # The selected DVGT layers are single-scale patch grids. Expose a DPT
        # style pyramid relative to the configured patch grid instead of the
        # old 224x448-only constants.
        hp, wp = self.patch_grid
        f1_size = (hp * 4, wp * 4)
        f2_size = (hp * 2, wp * 2)
        f4_size = (max(1, hp // 2), max(1, wp // 2))
        f1 = F.interpolate(projected[0], size=f1_size, mode="bilinear", align_corners=False)
        f2 = F.interpolate(projected[1], size=f2_size, mode="bilinear", align_corners=False)
        f3 = projected[2]
        f4 = F.interpolate(projected[3], size=f4_size, mode="bilinear", align_corners=False)
        return ReassembledFeatures(
            f1=self._unflatten(f1, aggregated_tokens[self.selected_layers[0]]),
            f2=self._unflatten(f2, aggregated_tokens[self.selected_layers[0]]),
            f3=self._unflatten(f3, aggregated_tokens[self.selected_layers[0]]),
            f4=self._unflatten(f4, aggregated_tokens[self.selected_layers[0]]),
        )

    def _project(self, tokens: torch.Tensor, raw_patch_tokens: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError(f"Expected tokens [B,T,V,N,C], got {tuple(tokens.shape)}")
        if raw_patch_tokens.ndim != 5:
            raise ValueError(f"Expected raw patch tokens [B,T,V,P,C], got {tuple(raw_patch_tokens.shape)}")
        b, t, v, n, c = tokens.shape
        hp, wp = self.patch_grid
        patch_count = hp * wp
        special_tokens = self.special_tokens if self.special_tokens is not None else n - patch_count
        if special_tokens < 0:
            raise ValueError(f"Token count {n} is smaller than expected patch count {patch_count}")
        patch_tokens = tokens[:, :, :, special_tokens:, :]
        if patch_tokens.shape[3] != hp * wp:
            raise ValueError(
                f"Expected {hp * wp} patch tokens after {special_tokens} special tokens, "
                f"got {patch_tokens.shape[3]}"
            )
        if raw_patch_tokens.shape[:4] != (b, t, v, patch_count):
            raise ValueError(
                "Raw patch tokens must align with aggregated patch tokens; "
                f"expected {(b, t, v, patch_count, raw_patch_tokens.shape[-1])}, got {tuple(raw_patch_tokens.shape)}"
            )
        joint_tokens = torch.cat([patch_tokens, raw_patch_tokens], dim=-1)
        x = joint_tokens.reshape(b * t * v, hp, wp, joint_tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()
        return projection(x)

    @staticmethod
    def _unflatten(x: torch.Tensor, reference_tokens: torch.Tensor) -> torch.Tensor:
        b, t, v = reference_tokens.shape[:3]
        c, h, w = x.shape[1:]
        return x.reshape(b, t, v, c, h, w)
