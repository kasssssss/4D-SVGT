import torch
import torch.nn as nn
from typing import Dict, Optional
from iopath.common.file_io import g_pathmgr
from einops import rearrange
import logging
from hydra.utils import instantiate
from safetensors.torch import load_file

from dvgt.models.backbones.dvgt2_aggregator import DVGT2Aggregator
from dvgt.models.heads.dpt_head import DPTHead
from dvgt.utils.pose_encoding import decode_pose
from dvgt.evaluation.utils.geometry import accumulate_transform_points_and_pose_to_first_frame
from dvgt.utils.general import merge_stream_outputs

class DVGT2(nn.Module):
    def __init__(
        self, 
        patch_size=16, 
        embed_dim=1024,
        depth=24, 
        patch_embed="dinov3_vitl16",
        dino_v3_weight_path="ckpt/dino_v3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        enable_point=True, 
        frames_chunk_size=None, # DPT head: Number of frames to process in each chunk.
        ego_pose_head_conf=None,
        future_frame_window: int = 8,
        relative_pose_window: int = 1,
        use_causal_mask: bool = True,
    ):
        super().__init__()

        self.frames_chunk_size = frames_chunk_size

        dim_in = 3 * embed_dim

        self.future_frame_window = future_frame_window
        self.relative_pose_window = relative_pose_window
        self.ego_pose_window = self.future_frame_window + self.relative_pose_window

        self.aggregator = DVGT2Aggregator(
            patch_size=patch_size, embed_dim=embed_dim, depth=depth, 
            patch_embed=patch_embed, dino_v3_weight_path=dino_v3_weight_path,
            use_causal_mask=use_causal_mask, relative_pose_window=self.relative_pose_window,
            future_frame_window=self.future_frame_window,
        )

        self.ego_pose_head = instantiate(ego_pose_head_conf, dim_in=dim_in) if ego_pose_head_conf else None
        self.point_head = DPTHead(dim_in=dim_in, patch_size=patch_size, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None

    def forward(self, images: torch.Tensor, ego_status: Optional[torch.Tensor] = None):
        """
        Forward pass of the DVGT model.

        Args:
            images (torch.Tensor): Input images with shape [T, V, 3, H, W] or [B, T, V, 3, H, W], in range [0, 1].
                B: batch size, T: num_frames, V: views_per_frame, 3: RGB channels, H: height, W: width

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Ego pose encoding with shape [B, T, V, 7] (from the last iteration)
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, T, V, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, T, V, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """        
        # If without batch dimension, add it
        if len(images.shape) == 5:
            images = images.unsqueeze(0)
            
        aggregated_tokens_list, patch_start_idx, _ = self.aggregator(images)

        predictions = {}

        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            if self.ego_pose_head is not None:
                # Use tokens from the last block for ego pose prediction.
                ego_tokens = aggregated_tokens_list[-1]
                ego_tokens = ego_tokens[:, :, :, :self.ego_pose_window]
                ego_pose_outputs = self.ego_pose_head(ego_tokens, ego_status)
                predictions.update(ego_pose_outputs)

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, frames_chunk_size=self.frames_chunk_size
                )
                predictions["points"] = pts3d
                predictions["points_conf"] = pts3d_conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    def inference(self, images: torch.Tensor, ego_status: Optional[torch.Tensor] = None, infer_window: int = 4):
        """
        Streaming inference pass of the Stream DVGT model.

        Args:
            images (List[torch.Tensor]): Input images with shape T * [B, V, 3, H, W], or T * [V, 3, H, W], in range [0, 1].
                B: batch size, T: num_frames, V: views_per_frame, 3: RGB channels, H: height, W: width
        
        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Ego pose encoding with shape [B, T, V, 7] (from the last iteration)
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, T, V, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, T, V, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """        
        past_key_values = None
        ego_token_cache = None

        if len(images.shape) == 5: # Add Batch dim
            images = images.unsqueeze(0)
            if ego_status is not None:
                ego_status = ego_status.unsqueeze(0)    # [B, T, 8], vel, acc, command

        T = images.shape[1]
        ego_stat = None

        outputs_list = []

        for t in range(T):
            image = images[:, t : t+1]
        
            if ego_status is not None:
                start_idx = max(0, t - infer_window + 1)
                ego_stat = ego_status[:, start_idx : t + 1]

            if t >= infer_window and past_key_values is not None:
                update_past_key_values = []
                for past_k, past_v in past_key_values:
                    update_past_k = rearrange(past_k, 'v head (t p) d -> v head t p d', t=infer_window)
                    update_past_v = rearrange(past_v, 'v head (t p) d -> v head t p d', t=infer_window)
                    update_past_key_values.append([
                        update_past_k[:, :, -infer_window + 1:].flatten(2, 3),
                        update_past_v[:, :, -infer_window + 1:].flatten(2, 3)
                    ])
                    past_key_values = update_past_key_values

            aggregated_tokens_list, patch_start_idx, past_key_values = self.aggregator(
                image, 
                past_key_values=past_key_values, 
                use_cache=True, 
                past_frame_idx=t
            )

            predictions = {}

            with torch.amp.autocast(device_type=image.device.type, enabled=False):
                if self.ego_pose_head is not None:
                    curr_ego_tokens = aggregated_tokens_list[-1]
                    curr_ego_tokens = curr_ego_tokens[:, :, :, :self.ego_pose_window]
                    if (ego_token_cache is None) or (infer_window == 1):
                        ego_token_cache = curr_ego_tokens
                    else:
                        ego_token_cache = torch.cat((ego_token_cache[:, -infer_window+1:], curr_ego_tokens), dim=1)
                    ego_pose_outputs = self.ego_pose_head(ego_token_cache, ego_stat)
                    predictions.update(ego_pose_outputs)

                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens_list, images=image, patch_start_idx=patch_start_idx, frames_chunk_size=self.frames_chunk_size
                    )
                    predictions["points"] = pts3d
                    predictions["points_conf"] = pts3d_conf

            if not self.training:
                predictions["images"] = image  # store the images for visualization during inference
            
            outputs_list.append(predictions)
        
        # merge T dim
        final_outputs = merge_stream_outputs(outputs_list)
        return final_outputs

    def custom_load_state_dict(self, checkpoint_conf: Dict):
        logging.info(f"Loading checkpoint from {checkpoint_conf}")

        if checkpoint_conf.get('direct_load_pretrained_weights_path', None):
            with g_pathmgr.open(checkpoint_conf['direct_load_pretrained_weights_path'], "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")

            model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            missing, unexpected = self.load_state_dict(model_state_dict, strict=False)
            return missing, unexpected, None

        elif checkpoint_conf.get('load_pretrained_weights_path_vggt', None):
            state_dict = load_file(checkpoint_conf['load_pretrained_weights_path_vggt'], device="cpu")
            new_state_dict = {}
            for name, params in state_dict.items():
                # Rule 1: Rename frame_blocks to intra_view_blocks.
                if 'aggregator.frame_blocks' in name:
                    new_name = name.replace('frame_blocks', 'intra_view_blocks')
                    new_state_dict[new_name] = params

                # Rule 2: Copy the parameters of the global_blocks to other block types according to the aa_order configuration.
                elif 'aggregator.global_blocks' in name:
                    new_name = name.replace('global_blocks', 'cross_view_blocks')
                    new_state_dict[new_name] = params
                    new_name = name.replace('global_blocks', 'cross_frame_blocks')
                    new_state_dict[new_name] = params

                # We do not load the decoder, force model to relearn from scratch.

            missing, unexpected = self.load_state_dict(
                new_state_dict, strict=checkpoint_conf.get('strict', True)
            )
            return missing, unexpected, None

        elif checkpoint_conf.get('load_pretrained_weights_path_dvgt', None):
            with g_pathmgr.open(checkpoint_conf['load_pretrained_weights_path_dvgt'], "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")

            model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            update_ckpt = {}

            for k, v in model_state_dict.items():
                if 'aggregator.ego_pose_token' == k:
                    future_ego_token = torch.empty(1, 8, 1024)
                    future_ego_token.normal_(mean=0.0, std=1e-6)
                    print(future_ego_token.std())
                    v = torch.cat([
                        v, future_ego_token
                    ], dim=1)
                    print(v.shape)
                elif 'ego_pose_head' in k:
                    continue
                update_ckpt[k] = v


            missing, unexpected = self.load_state_dict(
                update_ckpt, strict=checkpoint_conf.get('strict', True)
            )
            return missing, unexpected, None

        elif checkpoint_conf.get('resume_checkpoint_path', None):
            ckpt_path = checkpoint_conf['resume_checkpoint_path']

            with g_pathmgr.open(ckpt_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            
            # Load model state
            model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            missing, unexpected = self.load_state_dict(
                model_state_dict, strict=True
            )
            return missing, unexpected, checkpoint

        else:
            print("❕❕❕[Important] not load pretrained weights")
            return None, None, None

    def post_process_for_eval(
        self,
        predictions: Dict, 
        batch: Dict, 
        **kwargs,
    ) -> Dict:
        outputs = {}

        if kwargs.get("eval_traj", True):
            batch, outputs = self.ego_pose_head.post_process_traj_for_eval(batch, predictions, outputs)

        if kwargs.get('eval_point_and_pose', True):
            relative_ego_pose_enc = self.ego_pose_head.post_process_pose_for_eval(predictions)
            ego_past_to_ego_curr, _ = decode_pose(relative_ego_pose_enc, pose_encoding_type="absT_quaR")
            outputs['ego_n_to_ego_first'], outputs['points_in_ego_first'] = accumulate_transform_points_and_pose_to_first_frame(ego_past_to_ego_curr, predictions['points'])
            batch['ego_n_to_ego_first'], batch['points_in_ego_first'] = accumulate_transform_points_and_pose_to_first_frame(batch['ego_past_to_ego_curr'], batch['points_in_ego_n'])
            outputs['ray_depth'] = predictions['points'].norm(dim=-1, p=2)
            outputs['points_in_ego_first_conf'] = predictions['points_conf']

        # gt_scale_factor
        if kwargs.get("gt_scale_factor", 1.0) != 1.0:
            s = kwargs['gt_scale_factor']
            keys_to_scale_content = {
                'points_in_ego_first',
                'ray_depth',
            }
            keys_to_scale_translation = {
                'ego_n_to_ego_first'
            }
            for key in outputs.keys():
                if key in keys_to_scale_content:
                    outputs[key] /= s
                    batch[key] /= s
                elif key in keys_to_scale_translation:
                    outputs[key][..., :3, 3] /= s
                    batch[key][..., :3, 3] /= s
                elif key == 'trajectories':
                    outputs[key][..., :2] /= s
                    batch[key][..., :2] /= s

        return outputs
