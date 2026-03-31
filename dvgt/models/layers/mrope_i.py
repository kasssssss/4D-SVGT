# Inspired by:
#   https://github.com/JJJYmmm/Multimodal-RoPEs.git

from typing import Tuple
import torch
from einops import rearrange
import torch.nn as nn

def get_mrope_interleave_index(video_grid_thw: Tuple[int], device, T_start: int = 0) -> torch.Tensor:
    """
    inplementation of M RoPE Interleave
    Input: 
        video_grid_thw (Tuple): Shape [B, T, V, H, W]
        T_start (int): start time index for KV Cache
    Output: position_ids (Tensor): Shape [3, B*V, Seq_Len]
    """
    b, t, v, h, w = video_grid_thw

    # T 维度: [0, 1, 2, ..., t-1]
    t_index = (
        torch.arange(T_start, T_start+t, device=device)
        .view(-1, 1) # [T, 1]
        .expand(-1, h * w) # [T, H*W]
        .flatten() # [T*H*W]
    )

    # H 维度: [0, 0, ..., 1, 1, ...]
    h_index = (
        torch.arange(h, device=device)
        .view(1, -1, 1) # [1, H, 1]
        .expand(t, -1, w) # [T, H, W]
        .flatten() # [T*H*W]
    )

    # W 维度: [0, 1, ..., 0, 1, ...]
    w_index = (
        torch.arange(w, device=device)
        .view(1, 1, -1) # [1, 1, W]
        .expand(t, h, -1) # [T, H, W]
        .flatten() # [T*H*W]
    )

    # Spatial Reset
    mm_pos_ids = torch.stack([t_index, h_index, w_index]) # Shape: [3, Seq_Len]

    # 我们不区分单帧不同视角
    position_ids = mm_pos_ids.unsqueeze(1).expand(-1, b * v, -1)
    
    return rearrange(position_ids, 'd (b v) (t h w) -> d (b t v) (h w)', b=b, t=t, v=v, h=h, w=w).clone()

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MRopeInterleaveEmbedding(nn.Module):
    def __init__(self, head_dim=64, mrope_section=[12, 10, 10], base=1000.0):
        super().__init__()
        self.head_dim = head_dim
        self.mrope_section = mrope_section

        assert sum(self.mrope_section) == self.head_dim // 2, \
            f"mrope_section sum ({sum(self.mrope_section)}) must equal head_dim // 2 ({self.head_dim // 2})"
    
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _compute_frequency_components(self, x, position_ids):
        # position_ids: [3, bs, seq_len]
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions)

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            freqs = self.apply_transformation(
                freqs, self.mrope_section
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_transformation(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0].clone()  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, q, k, position_ids):
        """
        Args:
            q: [Batch, n_heads, Seq_Len, Head_Dim] (假设 unsqueeze_dim=1)
            k: [Batch, n_heads, Seq_Len, Head_Dim]
            position_ids: [3, Batch, Seq_Len] - 支持任意 Batch/Seq 维度组合
        """
        cos, sin = self._compute_frequency_components(q, position_ids)

        return apply_rotary_pos_emb(q, k, cos, sin)