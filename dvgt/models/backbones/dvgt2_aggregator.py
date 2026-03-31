import copy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple, List, Optional
from einops import rearrange

from dvgt.models.layers.block import Block
from dvgt.models.layers.attention import AttentionMRoPE
from dvgt.models.layers.mrope_i import MRopeInterleaveEmbedding, get_mrope_interleave_index

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class DVGT2Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in DVGT: Visual Geometry Transformer for autonomous Driving.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        patch_size (int): Patch size for image backbone.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "dinov3_vitl16".
        dino_v3_weight_path (str): Weight path for dinov3 corresponding to the patch embed.
        aa_order (list[str]): The order of alternating attention, e.g. ["intra_view", "cross_view", "cross_frame"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        init_values (float): Init scale for layer scale.
        max_frames (int): Maximum number of frames supported by model.
        use_causal_mask (bool): Whether to use causal mask for attention.
    """

    def __init__(
        self,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov3_vitl16",
        dino_v3_weight_path="ckpt/dino_v3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        aa_order=["intra_view", "cross_view", "cross_frame"],
        aa_block_size=1,
        qk_norm=True,
        init_values=0.01,
        relative_pose_window=1,
        future_frame_window=8,
        max_frames= 48,
        use_causal_mask=False,
    ):
        super().__init__()

        self.aa_order = aa_order
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        assert aa_block_size == 1, "aa_block_size must be 1 for now"
        self.aa_block_size = aa_block_size
        self.use_causal_mask = use_causal_mask

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")
        self.aa_block_num = self.depth // self.aa_block_size
        
        if dino_v3_weight_path:
            self.patch_embed = torch.hub.load('third_party/dinov3', patch_embed, source='local', weights=dino_v3_weight_path)   # Load model code and weight
        else:
            self.patch_embed = torch.hub.load('third_party/dinov3', patch_embed, source='local')   # Only load model code
        # Disable gradient updates for mask token
        if hasattr(self.patch_embed, "mask_token"):
            self.patch_embed.mask_token.requires_grad_(False)

        # Initialize rotary position embedding
        self.rope = MRopeInterleaveEmbedding()
        self.position_getter = get_mrope_interleave_index

        self.intra_view_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    attn_class=AttentionMRoPE,
                )
                for _ in range(depth)
            ]
        )
        self.cross_view_blocks = copy.deepcopy(self.intra_view_blocks)
        self.cross_frame_blocks = copy.deepcopy(self.intra_view_blocks)

        # The patch tokens start after the ego, camera and register tokens
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, num_register_tokens, embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        ego_pose_token_num_per_image = future_frame_window + relative_pose_window
        self.ego_pose_token = nn.Parameter(torch.randn(1, ego_pose_token_num_per_image, embed_dim))
        # Initialize parameters with small values
        nn.init.normal_(self.ego_pose_token, std=1e-6)
        self.patch_start_idx += ego_pose_token_num_per_image

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 1, 3, 1, 1), persistent=False)

        if self.use_causal_mask:
            cached_time_mask = torch.tril(torch.ones(max_frames, max_frames, dtype=torch.bool))
            self.register_buffer('cached_time_mask', cached_time_mask, persistent=False)

        self.use_reentrant = False # hardcoded to False

    def forward(
        self, 
        images: torch.Tensor, 
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        past_frame_idx: int = 0,
    ) -> Tuple[List[torch.Tensor], int, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, T, V, 3, H, W], in range [0, 1].
                B: batch size, T: num_frames, V: views_per_frame, 3: RGB channels, H: height, W: width
            past_frame_idx: current stream index
        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, T, V, C_in, H, W = images.shape

        T_start = past_frame_idx if use_cache else 0

        if use_cache:
            assert T == 1, f"Use KV cache expects T=1, got T={T}"

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*T*V, C, H, W] for patch embedding
        images = images.view(B * T * V, C_in, H, W)
        patch_tokens = self.patch_embed.forward_features(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        # Special tokens
        register_token = self.register_token.expand(B * T * V, -1, -1).contiguous()
        ego_pose_token = self.ego_pose_token.expand(B * T * V, -1, -1).contiguous()
        tokens = [ego_pose_token, register_token, patch_tokens]
        tokens = torch.cat(tokens, dim=1)

        # m rope interleave: [3, b*t*v, h*w]
        pos = self.position_getter((B, T, V, H // self.patch_size, W // self.patch_size), images.device, T_start=T_start)

        if self.patch_start_idx > 0:
            # Shift existing spatial coordinates to reserve (H=0, W=0) for Special Tokens.
            # Temporal indices (T) remain unchanged to maintain cross-frame alignment.
            pos[1:] = pos[1:] + 1

            # Construct positional IDs for Special Tokens: [3, Batch*Time*View, patch_start_idx]
            # Since all tokens within a patch share the same temporal index, 
            # we anchor the Special Token's T-coordinate to the first token of each sequence.
            t_idx = pos[0, :, :1].clone()
            t_special = t_idx.expand(-1, self.patch_start_idx) # [N, self.patch_start_idx]
            hw_special = torch.zeros_like(t_special)
            special_pos = torch.stack([t_special, hw_special, hw_special])
            pos = torch.cat([special_pos, pos], dim=-1)

        # update P because we added special tokens
        _, P, C = tokens.shape
        intra_view_idx, cross_view_idx, cross_frame_idx = 0, 0, 0
        output_list = []
        next_key_values = [] if use_cache else None

        attn_mask = None
        if not use_cache and T > 1 and self.use_causal_mask:
            attn_mask = self.cached_time_mask[:T, :T]
            attn_mask = attn_mask.repeat_interleave(P, dim=0).repeat_interleave(P, dim=1)

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "intra_view":
                    tokens, intra_view_idx, intra_view_intermediates = self._process_intra_view_attention(
                        tokens, B, T, V, P, C, intra_view_idx, pos=pos
                    )
                elif attn_type == "cross_view":
                    tokens, cross_view_idx, cross_view_intermediates = self._process_cross_view_attention(
                        tokens, B, T, V, P, C, cross_view_idx, pos=pos
                    )
                elif attn_type == "cross_frame":
                    layer_past_kv = past_key_values[cross_frame_idx] if (use_cache and past_key_values is not None) else None

                    tokens, cross_frame_idx, cross_frame_intermediates, layer_new_kv = self._process_cross_frame_attention(
                        tokens, B, T, V, P, C, cross_frame_idx, pos=pos,
                        past_key_value=layer_past_kv,
                        use_cache=use_cache,
                        attn_mask=attn_mask
                    )

                    if use_cache:
                        next_key_values.append(layer_new_kv)

                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for j in range(len(intra_view_intermediates)):
                # concat intermediates, [B*T*V, P, 3C]
                concat_inter = [cross_frame_intermediates[j], cross_view_intermediates[j], intra_view_intermediates[j]]
                output_list.append(torch.cat(concat_inter, dim=-1))


        del concat_inter, intra_view_intermediates, cross_view_intermediates, cross_frame_intermediates

        return output_list, self.patch_start_idx, next_key_values

    def _process_intra_view_attention(self, tokens, B, T, V, P, C, start_idx, pos):
        # Shape: (B*T*V, P, C)
        # RoPE pos: (3, B*T*V, P) matches tokens perfectly
        intermediates = []
        idx = start_idx
        
        for _ in range(self.aa_block_size):
            blk = self.intra_view_blocks[idx]
            if self.training:
                tokens, _ = checkpoint(blk, tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens, _ = blk(tokens, pos=pos)
            
            idx += 1
            intermediates.append(tokens.view(B, T, V, P, C))
            
        return tokens, idx, intermediates

    def _process_cross_view_attention(self, tokens, B, T, V, P, C, start_idx, pos):
        # Target Shape: (B*T, V*P, C)
        # Reshape tokens
        tokens_reshaped = tokens.view(B * T, V * P, C)
        
        # Reshape pos if necessary
        pos_reshaped = None
        if pos is not None:
             pos_reshaped = pos.view(3, B * T, V * P)

        intermediates = []
        idx = start_idx
        
        for _ in range(self.aa_block_size):
            blk = self.cross_view_blocks[idx]
            if self.training:
                tokens_reshaped, _ = checkpoint(blk, tokens_reshaped, pos_reshaped, use_reentrant=self.use_reentrant)
            else:
                tokens_reshaped, _ = blk(tokens_reshaped, pos=pos_reshaped)
            
            idx += 1
            # Reshape back for intermediates storage: (B, T, V, P, C)
            intermediates.append(tokens_reshaped.view(B, T, V, P, C))
        
        # Return flattened (B*T*V, P, C) for next stage to maintain consistency
        return tokens_reshaped.view(B * T * V, P, C), idx, intermediates

    def _process_cross_frame_attention(
        self, tokens, B, T, V, P, C, start_idx, pos,
        past_key_value=None, use_cache=False, attn_mask=None,
    ):
        # Target Shape: (B*V, T*P, C)
        tokens_reshaped = tokens.view(B, T, V, P, C).transpose(1, 2).contiguous().view(B * V, T * P, C)
        
        pos_reshaped = None
        if pos is not None:
            pos_reshaped = rearrange(pos, 'd (b t v) p -> d (b v) (t p)', b=B, t=T, v=V, p=P)

        intermediates = []
        idx = start_idx
    
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            blk = self.cross_frame_blocks[idx]

            if self.training:
                tokens_reshaped, new_kv = checkpoint(
                    blk, 
                    tokens_reshaped, 
                    pos_reshaped, 
                    attn_mask=attn_mask, 
                    use_cache=False,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens_reshaped, new_kv = blk(
                    tokens_reshaped, 
                    pos=pos_reshaped, 
                    attn_mask=attn_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
                
            idx += 1
            inter_view = tokens_reshaped.view(B, V, T, P, C).transpose(1, 2).contiguous()
            intermediates.append(inter_view)

        # We need (B, T, V, P, C)
        tokens_out = tokens_reshaped.view(B, V, T, P, C).transpose(1, 2).contiguous().view(B * T * V, P, C)
        return tokens_out, idx, intermediates, new_kv