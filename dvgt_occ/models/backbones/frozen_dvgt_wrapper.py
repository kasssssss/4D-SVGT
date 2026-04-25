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

        aggregated_tokens_list, patch_start_idx, _, raw_patch_tokens = aggregator(images, return_patch_tokens=True)
        outputs: Dict[str, object] = {}
        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            ego_pose_head = getattr(self.dvgt_model, "ego_pose_head", None)
            if ego_pose_head is not None:
                ego_tokens = aggregated_tokens_list[-1]
                ego_window = int(getattr(self.dvgt_model, "ego_pose_window", ego_tokens.shape[3]))
                ego_tokens = ego_tokens[:, :, :, :ego_window]
                outputs.update(ego_pose_head(ego_tokens, ego_status))
                if outputs.get("relative_ego_pose_enc") is None and hasattr(ego_pose_head, "post_process_pose_for_eval"):
                    outputs["relative_ego_pose_enc"] = ego_pose_head.post_process_pose_for_eval(outputs)
            point_head = getattr(self.dvgt_model, "point_head", None)
            if point_head is not None:
                pts3d, pts3d_conf = point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=getattr(self.dvgt_model, "frames_chunk_size", None),
                )
                outputs["points"] = pts3d
                outputs["points_conf"] = pts3d_conf
        selected = {layer: aggregated_tokens_list[layer] for layer in self.selected_layers}
        return {
            "aggregated_tokens": selected,
            "raw_patch_tokens": raw_patch_tokens,
            "patch_start_idx": patch_start_idx,
            "points": outputs.get("points"),
            "points_conf": outputs.get("points_conf"),
            "relative_ego_pose_enc": outputs.get("relative_ego_pose_enc"),
            "relative_ego_pose_enc_list": outputs.get("relative_ego_pose_enc_list"),
            "ego_pose": outputs.get("pose_enc", outputs.get("ego_pose", outputs.get("relative_ego_pose_enc"))),
        }
