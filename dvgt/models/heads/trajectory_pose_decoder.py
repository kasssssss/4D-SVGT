# Inspired by:
#   https://github.com/hustvl/diffusiondrive

from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import numpy as np
import copy
import math
from pathlib import Path
try:
    from diffusers.schedulers import DDIMScheduler
except ImportError:  # pragma: no cover - lightweight training containers may not ship diffusers
    class _DDIMStepOutput:
        def __init__(self, prev_sample):
            self.prev_sample = prev_sample

    class DDIMScheduler:
        def __init__(self, num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample"):
            self.num_train_timesteps = int(num_train_timesteps)
            self.prediction_type = prediction_type
            if beta_schedule == "scaled_linear":
                betas = torch.linspace(1e-4 ** 0.5, 2e-2 ** 0.5, self.num_train_timesteps, dtype=torch.float32) ** 2
            else:
                betas = torch.linspace(1e-4, 2e-2, self.num_train_timesteps, dtype=torch.float32)
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.long)
            self._step_size = 1

        def add_noise(self, original_samples, noise, timesteps):
            alphas = self.alphas_cumprod.to(original_samples.device)[timesteps].to(original_samples.dtype)
            while alphas.ndim < original_samples.ndim:
                alphas = alphas.unsqueeze(-1)
            return alphas.sqrt() * original_samples + (1.0 - alphas).sqrt() * noise

        def set_timesteps(self, num_inference_steps, device=None):
            step = max(self.num_train_timesteps // int(num_inference_steps), 1)
            self._step_size = step
            self.timesteps = torch.arange(self.num_train_timesteps - step, -1, -step, dtype=torch.long, device=device)

        def step(self, model_output, timestep, sample):
            if self.prediction_type != "sample":
                raise NotImplementedError("Fallback DDIMScheduler only supports prediction_type='sample'.")
            timestep_value = int(timestep.item() if torch.is_tensor(timestep) else timestep)
            prev_timestep = max(timestep_value - int(self._step_size), 0)
            alphas = self.alphas_cumprod.to(sample.device, dtype=sample.dtype)
            alpha_t = alphas[timestep_value]
            alpha_prev = alphas[prev_timestep] if prev_timestep >= 0 else sample.new_tensor(1.0)
            beta_t = (1.0 - alpha_t).clamp_min(1e-12)
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_t.sqrt() * pred_original_sample) / beta_t.sqrt()
            prev_sample = alpha_prev.sqrt() * pred_original_sample + (1.0 - alpha_prev).sqrt() * pred_epsilon
            return _DDIMStepOutput(prev_sample=prev_sample)

from dvgt.models.layers.block import CrossAttentionBlock

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    scale = 2 * math.pi
    dim_t = torch.arange(hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / hidden_dim)

    pos_embed = pos_tensor[..., None] * scale / dim_t
    pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1).flatten(-2)
    # We do not swap the dim of XYZ, model will handle it.
    return pos_embed.flatten(-2)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ModulationLayer(nn.Module):

    def __init__(
        self, 
        embed_dims: int, 
        condition_dims: int,
        if_zeroinit_reg=False,
    ):
        super().__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )

        self.if_zeroinit_reg = if_zeroinit_reg
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(self, x, time_embed):
        # x: [B, N, D], time_embed: [B, 1, D]
        scale_shift = self.scale_shift_mlp(time_embed)
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = x * (1 + scale) + shift
        return x

class DiffRefinementHead(nn.Module):

    def __init__(
        self,
        embed_dims=3096,
        output_shape=(8, 3),
        if_zeroinit_reg=False,
    ):
        super().__init__()
        self.output_shape = output_shape
        
        self.cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )

        output_dim = np.prod(output_shape)
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, output_dim),
        )
        self.if_zeroinit_reg = if_zeroinit_reg
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.reg_branch[-1].weight, 0)
            nn.init.constant_(self.reg_branch[-1].bias, 0)
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def forward(self, feature):
        # feature: [B, Mode, D]
        bs, mode, _ = feature.shape
        cls_score = self.cls_branch(feature).squeeze(-1) # [B, Mode]
        
        reg_delta = self.reg_branch(feature) # [B, Mode, OutDim]
        reg_delta = reg_delta.reshape(bs, mode, *self.output_shape)
            
        return reg_delta, cls_score

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        dim_in: int = 256,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        drop: float = 0.1,
        future_frame_window: int = 8,
        pose_output_dim: int = 7,
    ):
        super().__init__()
        self.traj_attn = CrossAttentionBlock(
            dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, 
            init_values=init_values, drop=drop
        )
        self.pose_attn = CrossAttentionBlock(
            dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, 
            init_values=init_values, drop=drop
        )
        
        self.traj_time_modulation = ModulationLayer(dim_in, dim_in)
        self.pose_time_modulation = ModulationLayer(dim_in, dim_in)

        self.traj_decoder = DiffRefinementHead(
            embed_dims=dim_in,
            output_shape=(future_frame_window, 3),
        )

        self.pose_decoder = DiffRefinementHead(
            embed_dims=dim_in,
            output_shape=(pose_output_dim,),
        )

    def forward(
        self, 
        traj_feature,
        pose_feature,
        past_latent_token,
        future_latent_token, 
        noisy_traj,
        noisy_pose_translation,
        time_embed,
    ):
        pose_feature = self.pose_attn(pose_feature, past_latent_token)
        traj_feature = self.traj_attn(traj_feature, future_latent_token)

        pose_feature = self.pose_time_modulation(pose_feature, time_embed)
        traj_feature = self.traj_time_modulation(traj_feature, time_embed)

        traj_reg, traj_cls = self.traj_decoder(traj_feature)
        poses_reg, poses_cls = self.pose_decoder(pose_feature)

        traj_reg[..., :2] = traj_reg[..., :2] + noisy_traj
        traj_reg[..., 2] = traj_reg[..., 2].tanh() * np.pi

        poses_reg[..., :3] = poses_reg[..., :3] + noisy_pose_translation
        
        return traj_reg, traj_cls, poses_reg, poses_cls

class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(
        self,
        pose_feature,
        traj_feature,
        past_latent_token,
        future_latent_token, 
        noisy_traj,
        noisy_pose_translation,
        time_embed,
    ):
        poses_reg_list = []
        poses_cls_list = []
        traj_reg_list = []
        traj_cls_list = []

        current_traj = noisy_traj
        current_pose_translation = noisy_pose_translation

        for blk in self.layers:
            traj_reg, traj_cls, poses_reg, poses_cls = blk(
                pose_feature=pose_feature,
                traj_feature=traj_feature,
                past_latent_token=past_latent_token,
                future_latent_token=future_latent_token,
                noisy_traj=current_traj,
                noisy_pose_translation=current_pose_translation,
                time_embed=time_embed,
            )

            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_reg_list.append(traj_reg)
            traj_cls_list.append(traj_cls)

            current_traj = traj_reg[..., :2].clone().detach()
            current_pose_translation = poses_reg[..., :3].clone().detach()

        return {
            "diff_traj_reg_list": traj_reg_list, 
            "diff_traj_cls_list": traj_cls_list, 
            "diff_poses_reg_list": poses_reg_list, 
            "diff_poses_cls_list": poses_cls_list
        }

class TrajectoryPoseDecoder(nn.Module):
    """
    Inspired by DiffusionDrive: Utilizing an iterative denoising process 
    to synthesize ego poses and future trajectories.
    """

    def __init__(
        self,
        dim_in: int = 256,
        num_layers: int = 2,
        traj_anchor_filepath: str = 'data_annotation/anchors/train_total_20/future_anchors_20_rdf.npz',
        pose_translation_anchor_filepath: str = 'data_annotation/anchors/train_total_20/past_anchors_20_rdf.npz',
        traj_max: List[float] = [33.7, 137.2],
        traj_min: List[float] = [-28.9, -7.4],
        pose_translation_max: List[float] = [1.5, 0.8, 1.0],
        pose_translation_min: List[float] = [-1.3, -0.7, -17.7],
        pose_encoding_type: str = "absT_quaR",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        future_frame_window: int = 8,
        past_frame_window: int = 1,
        drop: float = 0.1,
        truncated_noise_steps: int = 50,
        gt_scale_factor: float = 0.1,
        enable_ego_status: bool = False,
    ):
        super().__init__()
        self.truncated_noise_steps = truncated_noise_steps

        if pose_encoding_type == "absT_quaR":
            self.pose_output_dim = 7
        else:
            raise ValueError(f"Unsupported ego pose encoding type: {pose_encoding_type}")
        self.future_frame_window = future_frame_window
        self.past_frame_window = past_frame_window

        # Scheduler
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        # Traj Anchors
        if Path(traj_anchor_filepath).exists():
            traj_anchor_data = np.load(traj_anchor_filepath)
            traj_anchor = torch.from_numpy(traj_anchor_data['centers']).to(torch.float32)     # [N, 8, 2]
        else:
            traj_anchor = torch.zeros((20, future_frame_window, 2), dtype=torch.float32)
        traj_anchor = traj_anchor.unsqueeze(0) * gt_scale_factor  # [1, N, 8, 2]
        traj_max = torch.tensor(traj_max, dtype=torch.float32) * gt_scale_factor
        traj_min = torch.tensor(traj_min, dtype=torch.float32) * gt_scale_factor
        norm_traj_anchor = self.normalize(traj_anchor, traj_min, traj_max)
        self.register_buffer('norm_traj_anchor', norm_traj_anchor) 
        self.register_buffer('traj_max', traj_max) # x, y
        self.register_buffer('traj_min', traj_min)

        # ego status
        self.hist_encoding = None
        if enable_ego_status:
            self.hist_encoding = nn.Linear(8, dim_in)

        # Pose Translation Anchors
        if Path(pose_translation_anchor_filepath).exists():
            pose_anchor_data = np.load(pose_translation_anchor_filepath)
            pose_translation_anchor = torch.from_numpy(pose_anchor_data['centers']).to(torch.float32)     # [N, 3]
        else:
            pose_translation_anchor = torch.zeros((20, 3), dtype=torch.float32)
        pose_translation_anchor = pose_translation_anchor.unsqueeze(0) * gt_scale_factor  # [1, N, 3]
        pose_translation_max = torch.tensor(pose_translation_max, dtype=torch.float32) * gt_scale_factor
        pose_translation_min = torch.tensor(pose_translation_min, dtype=torch.float32) * gt_scale_factor
        norm_pose_translation_anchor = self.normalize(pose_translation_anchor, pose_translation_min, pose_translation_max)
        self.register_buffer('norm_pose_translation_anchor', norm_pose_translation_anchor)
        self.register_buffer('pose_translation_max', pose_translation_max)
        self.register_buffer('pose_translation_min', pose_translation_min)

        # 2. Embeddings
        self.traj_embedding_dim = dim_in * 2
        self.pose_embedding_dim = dim_in * 3

        self.traj_feature_proj = nn.Sequential(
            *linear_relu_ln(dim_in, 1, 1, self.traj_embedding_dim),
            nn.Linear(dim_in, dim_in),
        )
        self.pose_feature_proj = nn.Sequential(
            *linear_relu_ln(dim_in, 1, 1, self.pose_embedding_dim),
            nn.Linear(dim_in, dim_in),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim_in),
            nn.Linear(dim_in, dim_in * 4),
            nn.Mish(),
            nn.Linear(dim_in * 4, dim_in),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            future_frame_window=future_frame_window,
            dim_in=dim_in,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            drop=drop,
            pose_output_dim=self.pose_output_dim,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, num_layers)

    def normalize(self, x, x_min, x_max):
        # x: [..., D], min: [D]
        return 2 * (x - x_min) / (x_max - x_min + 1e-6) - 1

    def denormalize(self, x, x_min, x_max):
        return (x + 1) / 2 * (x_max - x_min + 1e-6) + x_min

    def forward(self, ego_token, ego_status):
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_token, ego_status)
        else:
            return self.forward_test(ego_token, ego_status)

    def get_feature_embedding(self, noisy_traj_phys, noisy_pose_phys):
        # traj
        traj_embed = gen_sineembed_for_position(noisy_traj_phys, hidden_dim=self.traj_embedding_dim // 2 // self.future_frame_window)
        traj_embed = traj_embed.flatten(-2)
        traj_feature = self.traj_feature_proj(traj_embed)

        # pose translation 
        pose_embed = gen_sineembed_for_position(noisy_pose_phys, hidden_dim=self.pose_embedding_dim // 3)
        pose_feature = self.pose_feature_proj(pose_embed)

        return traj_feature, pose_feature

    def recover_t_dim(self, outputs: Dict, bs: int):
        reshape_output = {}
        for key, values in outputs.items():
            reshape_output[key] = [v.view(bs, -1, *v.shape[1:]) for v in values]

        return reshape_output

    def forward_train(
        self, 
        ego_token: torch.Tensor, 
        ego_status: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        B, T, V, P, D = ego_token.shape
        super_B = B * T
        device = ego_token.device
        past_latent_token = ego_token[..., :self.past_frame_window, :].reshape(super_B, -1, D)
        future_latent_token = ego_token[..., self.past_frame_window:, :].reshape(super_B, -1, D)

        # Only 50 steps of shallow noise are injected to preserve
        timesteps = torch.randint(0, self.truncated_noise_steps, (super_B,), device=device).long()
        
        # trajectory
        norm_traj_anchor = self.norm_traj_anchor.repeat(super_B, 1, 1, 1)
        noise_traj = torch.randn_like(norm_traj_anchor)
        noisy_traj = self.diffusion_scheduler.add_noise(norm_traj_anchor, noise_traj, timesteps)
        noisy_traj = torch.clamp(noisy_traj, -1, 1)
        noisy_traj_phys = self.denormalize(noisy_traj, self.traj_min, self.traj_max)

        # pose
        norm_pose_anchor = self.norm_pose_translation_anchor.repeat(super_B, 1, 1)
        noise_pose = torch.randn_like(norm_pose_anchor)
        noisy_pose = self.diffusion_scheduler.add_noise(norm_pose_anchor, noise_pose, timesteps)
        noisy_pose = torch.clamp(noisy_pose, -1, 1)
        noisy_pose_phys = self.denormalize(noisy_pose, self.pose_translation_min, self.pose_translation_max)

        # Feature Embedding
        traj_feature, pose_feature = self.get_feature_embedding(noisy_traj_phys, noisy_pose_phys)

        if self.hist_encoding is not None:
            ego_status = ego_status.reshape(super_B, -1)
            traj_feature += self.hist_encoding(ego_status[:, None])

        time_embed = self.time_mlp(timesteps).unsqueeze(1) # [B, 1, D]

        outputs = self.diff_decoder(
            pose_feature=pose_feature,
            traj_feature=traj_feature,
            past_latent_token=past_latent_token,
            future_latent_token=future_latent_token,
            noisy_traj=noisy_traj_phys,
            noisy_pose_translation=noisy_pose_phys,
            time_embed=time_embed
        )

        return self.recover_t_dim(outputs, B)

    @torch.no_grad()
    def forward_test(self, ego_token, ego_status) -> Dict[str, torch.Tensor]:
        """
        Inference with Truncated Diffusion (Refining Anchors)
        """
        B, T, V, P, D = ego_token.shape
        super_B = B * T
        device = ego_token.device
        past_latent_token = ego_token[..., :self.past_frame_window, :].reshape(super_B, -1, D)
        future_latent_token = ego_token[..., self.past_frame_window:, :].reshape(super_B, -1, D)

        # Following DiffusionDrive's official strategy: 
        # Though trained on [0, 50) steps, inference starts from step 10 for efficiency.
        # Only 2 steps are performed (10 -> 0) to achieve rapid trajectory refinement.
        start_step = 10
        sample_interval = 10
        total_inference_steps_setting = int(1000 / sample_interval)
        self.diffusion_scheduler.set_timesteps(total_inference_steps_setting, device=device)
        all_timesteps = self.diffusion_scheduler.timesteps
        start_idx = (all_timesteps == start_step).nonzero(as_tuple=True)[0].item()
        timesteps = all_timesteps[start_idx:] 

        # add noisy
        init_t = torch.full((super_B,), int(timesteps[0]), device=device, dtype=torch.long)
        # Trajectory
        norm_traj_anchor = self.norm_traj_anchor.repeat(super_B, 1, 1, 1)
        noise_traj = torch.randn_like(norm_traj_anchor)
        noisy_traj = self.diffusion_scheduler.add_noise(norm_traj_anchor, noise_traj, init_t)
        # Pose
        norm_pose_anchor = self.norm_pose_translation_anchor.repeat(super_B, 1, 1)
        noise_pose = torch.randn_like(norm_pose_anchor)
        noisy_pose = self.diffusion_scheduler.add_noise(norm_pose_anchor, noise_pose, init_t)

        # Denoising Loop
        for i, t in enumerate(timesteps):
            noisy_traj = torch.clamp(noisy_traj, -1, 1)
            noisy_pose = torch.clamp(noisy_pose, -1, 1)
            
            # Denormalize
            noisy_traj_phys = self.denormalize(noisy_traj, self.traj_min, self.traj_max)
            noisy_pose_phys = self.denormalize(noisy_pose, self.pose_translation_min, self.pose_translation_max)
            
            # Feature embedding
            traj_feature, pose_feature = self.get_feature_embedding(noisy_traj_phys, noisy_pose_phys)

            if self.hist_encoding is not None:
                ego_status = ego_status.reshape(super_B, -1)
                traj_feature += self.hist_encoding(ego_status[:, None])

            t_batch = torch.full((super_B,), int(t), device=device, dtype=torch.long)
            time_embed = self.time_mlp(t_batch).unsqueeze(1)

            # Include key: traj_reg_list, traj_cls_list, poses_reg_list, poses_cls_list
            outputs = self.diff_decoder(
                pose_feature=pose_feature,
                traj_feature=traj_feature,
                past_latent_token=past_latent_token,
                future_latent_token=future_latent_token,
                noisy_traj=noisy_traj_phys,
                noisy_pose_translation=noisy_pose_phys,
                time_embed=time_embed
            )            
            pred_clean_traj_phys = outputs['diff_traj_reg_list'][-1][..., :2] # 取XY
            pred_clean_pose_phys = outputs['diff_poses_reg_list'][-1][..., :3] # 取Trans

            # DNormalize Prediction for Scheduler
            pred_clean_traj = self.normalize(pred_clean_traj_phys, self.traj_min, self.traj_max)
            pred_clean_pose = self.normalize(pred_clean_pose_phys, self.pose_translation_min, self.pose_translation_max)

            noisy_traj = self.diffusion_scheduler.step(
                model_output=pred_clean_traj,
                timestep=t,
                sample=noisy_traj
            ).prev_sample

            noisy_pose = self.diffusion_scheduler.step(
                model_output=pred_clean_pose,
                timestep=t,
                sample=noisy_pose
            ).prev_sample

        return self.recover_t_dim(outputs, B)
