"""Frozen DVGT frontend wrapper.

This wrapper exposes the contract needed by the new heads without changing the
upstream DVGT model code. It can be used online for sanity checks, while normal
training should consume the offline cache produced from the same fields.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class FrozenDVGTWrapper(nn.Module):
    def __init__(self, dvgt_model: nn.Module, selected_layers: Iterable[int] = (4, 11, 17, 23)) -> None:
        super().__init__()
        self.dvgt_model = dvgt_model
        self.selected_layers = tuple(selected_layers)
        self.dvgt_model.eval()
        for parameter in self.dvgt_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, ego_status: Optional[torch.Tensor] = None) -> Dict[str, object]:
        if images.ndim != 6:
            raise ValueError(f"Expected images [B,T,V,3,H,W], got {tuple(images.shape)}")

        aggregator = getattr(self.dvgt_model, "aggregator", None)
        if aggregator is None:
            raise AttributeError("The wrapped DVGT model must expose an 'aggregator' attribute.")

        aggregated_tokens_list, patch_start_idx, _ = aggregator(images)
        outputs = self.dvgt_model(images, ego_status=ego_status)
        selected = {layer: aggregated_tokens_list[layer] for layer in self.selected_layers}
        return {
            "aggregated_tokens": selected,
            "patch_start_idx": patch_start_idx,
            "points": outputs.get("points"),
            "points_conf": outputs.get("points_conf"),
            "ego_pose": outputs.get("pose_enc", outputs.get("ego_pose")),
        }
