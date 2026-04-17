"""Query-conditioned Gaussian refinement scaffold.

This stage is expected to consume ego0-space merged Gaussian candidates. The
explicit null/background branch for query-to-Gaussian routing lives upstream in
the later assignment module, so this scaffold keeps the refinement step narrow.
"""

import torch
import torch.nn as nn

from dvgt_occ.types import DynamicQueryOutput, EntityOutput, GaussianAssignmentOutput, GaussianOutput


class EntityAggregator(nn.Module):
    def __init__(self, instance_dim: int = 32, motion_dim: int = 16, query_dim: int = 256) -> None:
        super().__init__()
        self.query_to_inst = nn.Linear(query_dim, instance_dim)
        self.query_to_motion = nn.Linear(query_dim, motion_dim)

    def forward(
        self,
        gaussians: GaussianOutput,
        queries: DynamicQueryOutput,
        assignment: GaussianAssignmentOutput,
    ) -> EntityOutput:
        b, t, v, h, w = gaussians.center.shape[:5]
        num_queries = queries.query_feat.shape[2]

        query_inst = self.query_to_inst(queries.query_feat)
        query_motion = self.query_to_motion(queries.query_feat)

        assign_prob = assignment.assignment_prob.reshape(b, t, v, h, w, num_queries + 1)
        bg_prob = assign_prob[..., :1]
        fg_prob = assign_prob[..., 1:]
        fg_norm = fg_prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        query_inst_expand = query_inst[:, :, None, None, None, :, :]
        query_motion_expand = query_motion[:, :, None, None, None, :, :]
        pooled_inst = (fg_prob.unsqueeze(-1) * query_inst_expand).sum(dim=-2) / fg_norm
        pooled_motion = (fg_prob.unsqueeze(-1) * query_motion_expand).sum(dim=-2) / fg_norm

        refine_gate = 1.0 - bg_prob
        inst = gaussians.instance_affinity + refine_gate * pooled_inst
        motion = gaussians.motion_code + refine_gate * pooled_motion
        entity_id = assignment.assigned_query.reshape(b, t, v, h, w)
        return EntityOutput(entity_id=entity_id, refined_instance_affinity=inst, refined_motion_code=motion)
