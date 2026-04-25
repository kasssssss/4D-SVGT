"""Fixed loss-weight contracts for DVGT-Occ training stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class LossWeights:
    dyn_soft: float = 1.0
    presence: float = 0.5
    query_match: float = 0.5
    query_cls: float = 1.0
    query_box3d: float = 2.0
    track_mem_feat: float = 0.25
    track_mem_box: float = 0.5
    birth_death: float = 0.25
    occ: float = 1.0
    sem_occ: float = 1.0
    dyn_occ: float = 0.5
    gs_render: float = 1.0
    sky_alpha: float = 0.0
    inst_contrast: float = 0.2
    query2gs: float = 0.5
    q2gs_null: float = 0.25
    mask_render: float = 1.0
    occ_local2gs: float = 0.0
    gs2occ_local: float = 0.0
    sem_proj_2d: float = 0.0
    motion_cons: float = 0.0
    gs_sparse: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS = LossWeights()


def stage_c_bridge_warmup_weights() -> LossWeights:
    return LossWeights(
        occ_local2gs=0.05,
        gs2occ_local=0.05,
        sem_proj_2d=0.10,
        motion_cons=0.0,
        gs_sparse=0.0,
    )


def stage_b_after_stability_weights() -> LossWeights:
    return LossWeights(
        occ_local2gs=0.1,
        gs2occ_local=0.1,
        sem_proj_2d=0.25,
        motion_cons=0.1,
        gs_sparse=0.01,
    )


def merge_loss_weights(base: LossWeights, override: LossWeights) -> LossWeights:
    merged = base.to_dict()
    for key, value in override.to_dict().items():
        merged[key] = float(value)
    return LossWeights(**merged)


def resolve_loss_weights(
    step: int,
    base: LossWeights = DEFAULT_STAGE_B_SCAFFOLD_WEIGHTS,
    bridge_warmup: LossWeights | None = None,
    after_stability: LossWeights | None = None,
    bridge_start_step: int | None = None,
    stability_start_step: int | None = None,
) -> LossWeights:
    if bridge_warmup is not None and bridge_start_step is not None and step >= int(bridge_start_step):
        if after_stability is not None and stability_start_step is not None and step >= int(stability_start_step):
            return merge_loss_weights(base, after_stability)
        return merge_loss_weights(base, bridge_warmup)
    if after_stability is None or stability_start_step is None:
        return base
    if step < int(stability_start_step):
        return base
    return merge_loss_weights(base, after_stability)
