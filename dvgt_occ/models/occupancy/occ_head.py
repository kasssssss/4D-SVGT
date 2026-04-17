"""Occupancy head scaffold."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dvgt_occ.models.decoders import OccDenseDecoder
from dvgt_occ.types import DynamicDenseOutput, OccHeadOutput, ReassembledFeatures

from .lift_2d_to_3d import Lift2DTo3D


class OccHead(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        semantic_classes: int = 18,
        occ_shape_zyx=(10, 100, 100),
        x_range=(-40.0, 40.0),
        y_range=(-40.0, 40.0),
        z_range=(-2.0, 6.0),
        voxel_size: float = 0.8,
    ) -> None:
        super().__init__()
        self.decoder = OccDenseDecoder(channels=channels, full_channels=128)
        self.gradient_checkpointing = False
        self.lift = Lift2DTo3D(
            channels=channels,
            occ_shape_zyx=occ_shape_zyx,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            voxel_size=voxel_size,
        )
        self.trunk = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1), nn.GELU(), nn.Conv3d(channels, channels, 3, padding=1))
        self.occ_head = nn.Conv3d(channels, 1, 1)
        self.sem_head = nn.Conv3d(channels, semantic_classes, 1)
        self.dyn_head = nn.Conv3d(channels, 1, 1)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        features: ReassembledFeatures,
        dynamic: DynamicDenseOutput,
        points: torch.Tensor,
        points_conf: torch.Tensor,
    ) -> OccHeadOutput:
        if self.gradient_checkpointing and self.training:
            _, p2, full = checkpoint(self.decoder, *features.as_tuple(), use_reentrant=False)
        else:
            _, p2, full = self.decoder(*features.as_tuple())
        p2 = p2 + dynamic.dyn_feat_1_4
        volume = self.lift(p2, points=points, points_conf=points_conf)
        b, t, c, nz, ny, nx = volume.shape
        trunk = self.trunk(volume.reshape(b * t, c, nz, ny, nx)).reshape(b, t, c, nz, ny, nx)
        volumes = {
            "occ_1": trunk,
            "occ_2": self._downsample(trunk, 2),
            "occ_4": self._downsample(trunk, 4),
        }
        flat = trunk.reshape(b * t, c, nz, ny, nx)
        return OccHeadOutput(
            occ_volumes=volumes,
            occ_latent=trunk.mean(dim=(1, 3, 4, 5)),
            occ_logit=self.occ_head(flat).reshape(b, t, 1, nz, ny, nx),
            sem_logit=self.sem_head(flat).reshape(b, t, -1, nz, ny, nx),
            dyn_occ_logit=self.dyn_head(flat).reshape(b, t, 1, nz, ny, nx),
            aux_decoder_full=full,
        )

    @staticmethod
    def _downsample(volume: torch.Tensor, factor: int) -> torch.Tensor:
        b, t, c, nz, ny, nx = volume.shape
        y = F.avg_pool3d(volume.reshape(b * t, c, nz, ny, nx), kernel_size=factor, stride=factor, ceil_mode=True)
        return y.reshape(b, t, c, *y.shape[-3:])
