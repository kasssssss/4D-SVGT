"""Default constants for the DVGT-SAM3-DGGT-Occ project."""

from dataclasses import dataclass, field
from typing import Tuple


CAMERA_VIEW_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class OccGridConfig:
    x_range: Tuple[float, float] = (-40.0, 40.0)
    y_range: Tuple[float, float] = (-40.0, 40.0)
    z_range: Tuple[float, float] = (-2.0, 6.0)
    voxel_size: float = 0.8

    @property
    def shape_zyx(self) -> Tuple[int, int, int]:
        nx = int(round((self.x_range[1] - self.x_range[0]) / self.voxel_size))
        ny = int(round((self.y_range[1] - self.y_range[0]) / self.voxel_size))
        nz = int(round((self.z_range[1] - self.z_range[0]) / self.voxel_size))
        return nz, ny, nx


@dataclass(frozen=True)
class DVGTOccConfig:
    batch_size: int = 1
    num_frames: int = 8
    num_views: int = 6
    image_height: int = 224
    image_width: int = 448
    patch_size: int = 16
    token_dim: int = 1024
    agg_token_dim: int = 3072
    neck_dim: int = 256
    full_dim: int = 128
    dynamic_query_dim: int = 256
    instance_dim: int = 32
    motion_dim: int = 16
    dynamic_classes: int = 8
    semantic_classes: int = 18
    projected_semantic_classes: int = 9
    max_track_queries: int = 64
    new_queries: int = 64
    sparse_dynamic_anchors: int = 2048
    global_latents: int = 128
    occ_samples_per_scale: int = 8
    gs_bias_scale: float = 0.75
    render_splat_radius: int = 4
    selected_layers: Tuple[int, ...] = (4, 11, 17, 23)
    camera_view_ids: Tuple[int, ...] = CAMERA_VIEW_IDS
    occ_grid: OccGridConfig = field(default_factory=OccGridConfig)

    @property
    def patch_grid(self) -> Tuple[int, int]:
        return self.image_height // self.patch_size, self.image_width // self.patch_size

    @property
    def patch_tokens(self) -> int:
        hp, wp = self.patch_grid
        return hp * wp

    @property
    def total_queries(self) -> int:
        return self.max_track_queries + self.new_queries


DEFAULT_DVGT_OCC_CONFIG = DVGTOccConfig()
