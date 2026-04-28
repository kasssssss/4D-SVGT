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
    raw_patch_token_dim: int = 1024
    neck_dim: int = 256
    full_dim: int = 128
    dynamic_dense_dim: int = 128
    occ_lift_dim: int = 64
    occ_base_dim: int = 64
    occ_bottleneck_dim: int = 128
    gs_feature_dim: int = 128
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
    render_source_weight: float = 1.0
    render_heldout_weight: float = 0.1
    render_lpips_weight: float = 0.05
    render_loss_use_composite: bool = True
    render_loss_mask_sky: bool = False
    render_enable_sky_background: bool = True
    render_scale_max: float = 0.0
    render_rasterize_mode: str = "classic"
    render_keep_score_gate: float = 0.0
    render_keep_topk_ratio: float = 0.0
    render_antigrid_replicas: int = 1
    render_antigrid_jitter_m: float = 0.0
    gs_scale_reg_max: float = 0.0
    gs_scale_reg_weight: float = 0.0
    gs_scale_floor_target: float = 0.0
    gs_scale_floor_weight: float = 0.0
    gs_init_scale: float = 0.08
    gs_init_opacity_logit: float = -1.0
    gs_anchor_jitter_m: float = 0.0
    gs_output_level: str = "full"
    gs_scale_multiplier: float = 0.10
    source_render_own_view_only: bool = False
    sky_hidden_dim: int = 64
    sky_fourier_freqs: int = 6
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
    def joint_token_dim(self) -> int:
        return self.agg_token_dim + self.raw_patch_token_dim

    @property
    def total_queries(self) -> int:
        return self.max_track_queries + self.new_queries


DEFAULT_DVGT_OCC_CONFIG = DVGTOccConfig()
