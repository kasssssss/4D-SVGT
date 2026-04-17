"""Typed containers shared by the first DVGT-Occ scaffold."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class ClipManifestEntry:
    scene_id: str
    clip_id: str
    source_scene_dir: Path
    frame_ids: Sequence[int]
    view_ids: Sequence[int]
    image_paths: Mapping[int, Sequence[Path]]
    meta_path: Optional[Path]
    lidar_dir: Path
    intrinsics_dir: Path
    extrinsics_dir: Path
    lidar_pose_dir: Path
    cache_dir: Path
    supervision_dir: Path


@dataclass
class ReassembledFeatures:
    f1: torch.Tensor
    f2: torch.Tensor
    f3: torch.Tensor
    f4: torch.Tensor

    def as_tuple(self):
        return self.f1, self.f2, self.f3, self.f4


@dataclass
class DynamicDenseOutput:
    dyn_logit_1_4: torch.Tensor
    dyn_logit_full: torch.Tensor
    dyn_feat_1_4: torch.Tensor
    h2: torch.Tensor
    p2: torch.Tensor
    full: torch.Tensor


@dataclass
class OccHeadOutput:
    occ_volumes: Dict[str, torch.Tensor]
    occ_latent: torch.Tensor
    occ_logit: torch.Tensor
    sem_logit: torch.Tensor
    dyn_occ_logit: torch.Tensor
    aux_decoder_full: Optional[torch.Tensor] = None


@dataclass
class DynamicQueryOutput:
    presence_logit: torch.Tensor
    query_box3d: torch.Tensor
    query_cls_logit: torch.Tensor
    query_feat: torch.Tensor
    query_motion: torch.Tensor
    query_ref_points: Optional[torch.Tensor] = None
    track_query_mask: Optional[torch.Tensor] = None


@dataclass
class GaussianOutput:
    center: torch.Tensor
    offset: torch.Tensor
    opacity: torch.Tensor
    scale: torch.Tensor
    rotation: torch.Tensor
    feat_dc: torch.Tensor
    confidence: torch.Tensor
    instance_affinity: torch.Tensor
    motion_code: torch.Tensor
    aux_decoder_full: Optional[torch.Tensor] = None


@dataclass
class GaussianAssignmentOutput:
    assignment_prob: torch.Tensor
    assigned_query: torch.Tensor
    background_prob: torch.Tensor
    local_gate: torch.Tensor


@dataclass
class EntityOutput:
    entity_id: torch.Tensor
    refined_instance_affinity: torch.Tensor
    refined_motion_code: torch.Tensor
    gs_to_occ_feat: Optional[torch.Tensor] = None


@dataclass
class BridgeOutput:
    gs_to_occ_local: torch.Tensor
    occ_to_gs_local: torch.Tensor
    global_latents: torch.Tensor


@dataclass
class RenderOutput:
    render_alpha_static: torch.Tensor
    render_alpha_dynamic: torch.Tensor
    render_alpha_all: torch.Tensor
    sem_proj_2d: torch.Tensor
