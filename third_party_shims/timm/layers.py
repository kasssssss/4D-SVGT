"""Minimal local shim for timm.layers."""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return x


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        tensor.normal_(mean=mean, std=std)
        tensor.clamp_(min=a, max=b)
    return tensor
