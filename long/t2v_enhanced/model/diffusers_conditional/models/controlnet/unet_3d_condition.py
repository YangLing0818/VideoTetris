# Copyright 2023 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
# Copyright 2023 The ModelScope Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
# from diffusers.models.transformer_temporal import TransformerTemporalModel
from t2v_enhanced.model.diffusers_conditional.models.controlnet.transformer_temporal import TransformerTemporalModel
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
    transformer_g_c
)
from t2v_enhanced.model.diffusers_conditional.models.controlnet.conditioning import ConditionalModel
from einops import rearrange
from t2v_enhanced.model.layers.conv_channel_extension import Conv2D_ExtendedChannels
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet3DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    r"""
    UNet3DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, it will skip the normalization and activation layers in post-processing
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str] = (
            "UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 64,
        merging_mode: str = "addition",
        use_image_embedding: bool = False,
        use_fps_conditioning: bool = False,
        unet_params=None,
        use_reference_attention: bool = False
    ):
        super().__init__()
        channel_expansion = unet_params.use_of
        self.concat = unet_params.concat
        self.use_image_tokens = unet_params.use_image_tokens_main
        self.image_encoder_name = type(unet_params.image_encoder).__name__
        self.use_image_embedding = use_image_embedding
        self.sample_size = sample_size
        self.gradient_checkpointing = False
        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        # input
        conv_in_kernel = 3
        conv_out_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        ''' 
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )
        '''
        self.conv_in = Conv2D_ExtendedChannels(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding, in_channel_extension=5 if self.concat else 0,
        )
        if use_reference_attention:
            self.reference_conv_in = Conv2D_ExtendedChannels(
                in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
            )
            self.reference_proj = nn.Linear(block_out_channels[0], 1024)
        
        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )
        self.use_fps_conditioning = use_fps_conditioning
        if use_fps_conditioning:
            fps_embed_dim = block_out_channels[0] * 4
            fps_input_dim = block_out_channels[0]
            self.fps_embedding = TimestepEmbedding(
                fps_input_dim, fps_embed_dim, act_fn=act_fn)
            self.fps_proj = Timesteps(block_out_channels[0], True, 0)

        self.transformer_in = TransformerTemporalModel(
            num_attention_heads=8,
            attention_head_dim=attention_head_dim,
            in_channels=block_out_channels[0],
            num_layers=1,
        )

        # class embedding
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        self.merging_mode = merging_mode
        # print("self.merging_mode", self.merging_mode)
        if self.merging_mode.startswith("attention"):
            self.cross_attention_merger_down_blocks = nn.ModuleList([])
            self.cross_attention_merger_mid_block = nn.ModuleList([])
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=False,
                use_image_embedding=use_image_embedding,
                unet_params=unet_params,
            )
            self.down_blocks.append(down_block)

            if self.merging_mode.startswith("attention"):
                for idx in range(3):
                    self.cross_attention_merger_down_blocks.append(ConditionalModel(
                        input_channels=input_channel if idx == 0 else output_channel, conditional_model=self.merging_mode.split("attention_")[1]))

        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
            use_image_embedding=use_image_embedding,
            unet_params=unet_params,
        )
        if self.merging_mode.startswith("attention"):
            self.cross_attention_merger_mid_block = ConditionalModel(
                input_channels=block_out_channels[-1], conditional_model=self.merging_mode.split("attention_")[1])
        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(
                i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=False,
                use_image_embedding=use_image_embedding,
                unet_params=unet_params,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        '''
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )
        '''
        self.conv_out = Conv2D_ExtendedChannels(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding, out_channel_extension=2 if channel_expansion else 0,
        )

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * \
            [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(
                    f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, value=False):
        self.gradient_checkpointing = value
        self.mid_block.gradient_checkpointing = value
        for module in self.down_blocks + self.up_blocks:
            if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
                module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        reference_frames_features: Optional[torch.Tensor] = None,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, num_frames, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet3DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet3DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet3DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info(
                "Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        '''
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        '''
        debug = False
        if self.use_fps_conditioning:

            if torch.is_tensor(fps):
                assert (fps > -1).all(), "FPS not set"
                if len(fps.shape) == 0:
                    fps = fps[None].to(sample.device)
            else:
                assert (fps > -1), "FPS not set"
                is_mps = sample.device.type == "mps"
                if isinstance(fps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                fps = torch.tensor([fps], dtype=dtype, device=sample.device)
            fps = fps.expand(sample.shape[0])
            fps_proj = self.fps_proj(fps)
            fps_proj = fps_proj.to(dtype=self.dtype)
            fps_emb = self.fps_embedding(fps_proj)
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        num_frames = sample.shape[2]
        timesteps = timesteps.expand(sample.shape[0])
        batch_size = sample.shape[0]

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        emb = emb.repeat_interleave(repeats=num_frames, dim=0)
        if self.use_fps_conditioning:
            fps_emb = fps_emb.repeat_interleave(repeats=num_frames, dim=0)
            emb = emb + fps_emb

        if encoder_hidden_states is not None and not torch.is_tensor(encoder_hidden_states):
            tuple_encoder_hidden_states = encoder_hidden_states
            layout_masks = [tuple_encoder_hidden_states[i][0] for i in range(len(tuple_encoder_hidden_states))]
            encoder_hidden_states = torch.stack([tuple_encoder_hidden_states[i][1] for i in range(len(tuple_encoder_hidden_states))], dim=0)

        if not self.use_image_tokens and encoder_hidden_states.shape[1] > 77:
            encoder_hidden_states = encoder_hidden_states[:, :77]
        # print(f"MAIN with tokens = {encoder_hidden_states.shape[1]}")
        #if encoder_hidden_states.shape[1] > 77:
            # assert (
            #     encoder_hidden_states.shape[1]-77) % num_frames == 0, f"Encoder shape {encoder_hidden_states.shape}. Num frames = {num_frames}"
            # context_text, context_img = encoder_hidden_states[:,
            #                                                   :77, :], encoder_hidden_states[:, 77:, :]
            # context_text = context_text.repeat_interleave(
                # repeats=num_frames, dim=0)

            # if self.image_encoder_name == "FrozenOpenCLIPImageEmbedder":
            #     print("Using Open CLip")
            #     # context_img = context_img.repeat_interleave(
            #         # repeats=num_frames, dim=0)
            # else:
            #     assert False, "Not implemented"
            #     context_img = rearrange(
            #         context_img, 'b (t l) c -> (b t) l c', t=num_frames)

            # encoder_hidden_states = torch.cat(
            #     [context_text, context_img], dim=1)
        # else:
        #     encoder_hidden_states = encoder_hidden_states.repeat_interleave(
        #         repeats=num_frames, dim=0)

        # 2. pre-process
        sample = sample.permute(0, 2, 1, 3, 4).reshape(
            (sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = self.conv_in(sample)

        # if reference_frames_features is not None:
        #     reference_frames_features = reference_frames_features.repeat_interleave(
        #         repeats=int(encoder_hidden_states.shape[0]/reference_frames_features.shape[0]), dim=0)
        #     reference_frames_features = self.conv_in(reference_frames_features)
        #     reference_frames_features = rearrange(reference_frames_features, 'b c h w -> b (h w) c' )        
        # #
        if reference_frames_features is not None :
            reference_frames_features = reference_frames_features.repeat_interleave(
                repeats=int(encoder_hidden_states.shape[0]/reference_frames_features.shape[0]), dim=0)
            reference_frames_features = self.reference_conv_in(reference_frames_features)
            reference_frames_features = rearrange(reference_frames_features, 'b c h w -> b (h w) c' ) 
            reference_frames_features = self.reference_proj(reference_frames_features)       
        


        if num_frames > 1:
            if self.gradient_checkpointing:
                sample = transformer_g_c(
                    self.transformer_in, sample, num_frames)
            else:
                sample = self.transformer_in(
                    sample, num_frames=num_frames, attention_mask=attention_mask).sample        encoder_hidden_states = list(zip(layout_masks, encoder_hidden_states))
        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                    reference_frames_features=reference_frames_features,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, num_frames=num_frames)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            if self.merging_mode == "addition":
                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                ):
                    down_block_res_sample = down_block_res_sample + down_block_additional_residual
                    new_down_block_res_samples += (down_block_res_sample,)
            elif self.merging_mode.startswith("attention"):
                for down_block_res_sample, down_block_additional_residual, merger in zip(
                    down_block_res_samples, down_block_additional_residuals, self.cross_attention_merger_down_blocks
                ):

                    down_block_res_sample = merger(
                        rearrange(down_block_res_sample, "(B F) C H W -> B F C H W", B=batch_size), rearrange(down_block_additional_residual, "(B F) C H W -> B F C H W", B=batch_size))
                    down_block_res_sample = rearrange(
                        down_block_res_sample, "B F C H W -> (B F) C H W")
                    new_down_block_res_samples += (down_block_res_sample,)
            elif self.merging_mode == "overwrite":
                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                ):
                    down_block_res_sample = down_block_additional_residual
                    new_down_block_res_samples += (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                reference_frames_features=reference_frames_features,
            )

        if mid_block_additional_residual is not None:
            if self.merging_mode == "addition":
                sample = sample + mid_block_additional_residual
            elif self.merging_mode == "overwrite":
                sample = sample + mid_block_additional_residual
            elif self.merging_mode.startswith("attention"):
                sample = self.cross_attention_merger_mid_block(
                    rearrange(sample, "(B F) C H W -> B F C H W", B=batch_size), rearrange(mid_block_additional_residual, "(B F) C H W -> B F C H W", B=batch_size))
                sample = rearrange(sample, "B F C H W -> (B F) C H W")

        if debug:
            upblockout = (sample,)
        # 5. up
        # import pdb
        # pdb.set_trace()
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(
                upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample, output_states = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                    reference_frames_features=reference_frames_features,
                )
            else:
                sample, output_states = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )
            if debug:
                upblockout += output_states

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        # reshape to (batch, channel, framerate, width, height)
        sample = sample[None, :].reshape(
            (-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)
