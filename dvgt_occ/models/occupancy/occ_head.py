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
        dynamic_channels: int = 128,
        lift_channels: int = 64,
        bottleneck_channels: int = 128,
        occ_shape_zyx=(10, 100, 100),
        x_range=(-40.0, 40.0),
        y_range=(-40.0, 40.0),
        z_range=(-2.0, 6.0),
        voxel_size: float = 0.8,
        output_size: tuple[int, int] = (224, 448),
    ) -> None:
        super().__init__()
        self.decoder = OccDenseDecoder(channels=channels, full_channels=128, output_size=output_size)
        self.gradient_checkpointing = False
        self.lift = Lift2DTo3D(
            channels=lift_channels,
            occ_shape_zyx=occ_shape_zyx,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            voxel_size=voxel_size,
        )
        self.pre_lift = nn.Sequential(
            nn.Conv2d(channels + 3 + 1 + 1, lift_channels, 1),
            nn.GroupNorm(8, lift_channels),
            nn.SiLU(),
        )
        self.occ_conv1 = nn.Sequential(
            nn.Conv3d(lift_channels, lift_channels, 3, padding=1),
            nn.GELU(),
        )
        self.occ_down = nn.Sequential(
            nn.Conv3d(lift_channels, bottleneck_channels, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.occ_bottleneck = nn.Sequential(
            nn.Conv3d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.GELU(),
        )
        nz, ny, nx = occ_shape_zyx
        nz2, ny2, nx2 = ((nz + 1) // 2, (ny + 1) // 2, (nx + 1) // 2)
        self.pos_occ_1 = nn.Parameter(torch.zeros(1, lift_channels, nz, ny, nx))
        self.pos_occ_2 = nn.Parameter(torch.zeros(1, bottleneck_channels, nz2, ny2, nx2))
        self.pos_occ_b = nn.Parameter(torch.zeros(1, bottleneck_channels, nz2, ny2, nx2))
        self.skip_proj = nn.Conv3d(lift_channels, bottleneck_channels, 1)
        self.out_refine = nn.Sequential(
            nn.Conv3d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.GELU(),
        )
        self.occ_head = nn.Conv3d(bottleneck_channels, 1, 1)
        self.sem_head = nn.Conv3d(bottleneck_channels, semantic_classes, 1)
        self.dyn_head = nn.Conv3d(bottleneck_channels, 1, 1)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        features: ReassembledFeatures,
        dynamic: DynamicDenseOutput,
        xyz_1_4: torch.Tensor,
        conf_1_4: torch.Tensor,
    ) -> OccHeadOutput:
        if self.gradient_checkpointing and self.training:
            _, p2, full = checkpoint(self.decoder, *features.as_tuple(), use_reentrant=False)
        else:
            _, p2, full = self.decoder(*features.as_tuple())
        o_in = torch.cat([p2, xyz_1_4, conf_1_4, dynamic.dyn_logit_1_4], dim=3)
        b, t, v, _, h, w = o_in.shape
        o_feat2d = self.pre_lift(o_in.reshape(b * t * v, o_in.shape[3], h, w)).reshape(b, t, v, -1, h, w)
        volume = self.lift(o_feat2d, xyz_1_4=xyz_1_4, conf_1_4=conf_1_4)
        b, t, c, nz, ny, nx = volume.shape
        vol1 = self.occ_conv1(volume.reshape(b * t, c, nz, ny, nx)).reshape(b, t, c, nz, ny, nx)
        vol2_flat = self.occ_down(vol1.reshape(b * t, c, nz, ny, nx))
        vol2 = vol2_flat.reshape(b, t, vol2_flat.shape[1], *vol2_flat.shape[-3:])
        bt2 = b * t
        volb = self.occ_bottleneck(vol2_flat).reshape(b, t, vol2.shape[2], *vol2.shape[-3:])
        vol1_pe = vol1 + self.pos_occ_1.unsqueeze(1).to(vol1.dtype)
        vol2_pe = vol2 + self.pos_occ_2.unsqueeze(1).to(vol2.dtype)
        volb_pe = volb + self.pos_occ_b.unsqueeze(1).to(volb.dtype)
        up = F.interpolate(volb.reshape(bt2, volb.shape[2], *volb.shape[-3:]), size=vol1.shape[-3:], mode="trilinear", align_corners=False)
        skip = self.skip_proj(vol1.reshape(b * t, vol1.shape[2], *vol1.shape[-3:]))
        vout = self.out_refine(up + skip).reshape(b, t, -1, nz, ny, nx)
        volumes = {
            "occ_1": vol1_pe,
            "occ_2": vol2_pe,
            "occ_b": volb_pe,
        }
        flat = vout.reshape(b * t, vout.shape[2], nz, ny, nx)
        return OccHeadOutput(
            occ_volumes=volumes,
            occ_latent=volb.mean(dim=(3, 4, 5)),
            occ_logit=self.occ_head(flat).reshape(b, t, 1, nz, ny, nx),
            sem_logit=self.sem_head(flat).reshape(b, t, -1, nz, ny, nx),
            dyn_occ_logit=self.dyn_head(flat).reshape(b, t, 1, nz, ny, nx),
            aux_decoder_full=full,
        )
