import dataclasses
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import List, Union, Literal, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, LCMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler, Transformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, deprecate, unscale_lora_layers
from einops import rearrange
from lpips import lpips
from torch import nn
from torch.nn import init
from torchvision.ops import RoIAlign, roi_align

from unhcv.common import CfgNode
from unhcv.common.image import resize_coordinate
from unhcv.common.types import default_factory, PathStr, ModelOutput
from unhcv.nn.utils import ReplaceModule, load_checkpoint

__all__ = ['EntityGAN', 'EntityGANConfig', 'EntityGANOutput', 'GanTransformers']


@dataclasses.dataclass
class UNet2DConditionGANOutput(UNet2DConditionOutput):
    features: Optional[List[torch.Tensor]] = None


class UNet2DConditionModelGAN:

    @classmethod
    def init(cls, pretrained_model_name_or_path=None, model=None):
        if model is None:
            model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path)
        ReplaceModule.setattr_function(model, "forward", cls.forward)
        return model

    @staticmethod
    # from 3.42
    def forward(
        self: UNet2DConditionModel,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_intermediate: bool = False,
        return_up_features: bool = False,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        if return_up_features:
            features = []
        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

            # for flash diffusion
            if return_intermediate:
                if not return_dict:
                    return (sample,)
                return UNet2DConditionOutput(sample=sample)
            if return_up_features:
                features.append(sample)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
            if return_up_features:
                features.append(sample)
        if return_up_features:
            return UNet2DConditionGANOutput(features=features)

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionGANOutput(sample=sample)


@dataclass
class EntityGANConfig(CfgNode):
    gan_model_pretrained_path: PathStr = "model/stable-diffusion-v1-5/unet"
    use_up_features: bool = True
    # roi
    output_size: int = 14
    roi_align_config: dict = default_factory(dict(output_size=(14, 14)))
    pool_method: Literal['mean', ] = 'mean'
    features_in_channels: list = default_factory([1280, 1280, 1280, 640, 320])
    roi_in_channels: int = 320
    heads: int = 8
    num_transformer_blocks: int = 3


@dataclass
class EntityGANOutput(ModelOutput):
    prob: torch.Tensor = None
    roi_feature_last: torch.Tensor = None


class GanTransformers(nn.Module):
    def __init__(self, in_channels, num_transformer_blocks, heads, token_num):
        super().__init__()
        self.transformers = nn.Sequential(*[
            BasicTransformerBlock(in_channels, num_attention_heads=heads,
                                  attention_head_dim=in_channels // heads, activation_fn='gelu') for _ in
            range(num_transformer_blocks)])
        self.position_embedding = nn.Parameter(
            torch.randn(1, token_num, in_channels) / in_channels ** 0.5
        )
        self.linear_prob = nn.Sequential(nn.LayerNorm(in_channels), nn.SiLU(), nn.Linear(in_channels, 1))

    def forward(self, encoder_hidden_states):
        encoder_hidden_states = encoder_hidden_states + self.position_embedding
        encoder_hidden_states = self.transformers(encoder_hidden_states)
        prob = self.linear_prob(encoder_hidden_states[:, 0])
        return prob, encoder_hidden_states


class EntityGAN(nn.Module):
    def __init__(self, config=None, backbone=None):
        super().__init__()
        self.config = config = EntityGANConfig.from_other(config)
        if backbone is not None:
            backbone = UNet2DConditionModelGAN.init(config.gan_model_pretrained_path, model=backbone)
        self.backbone = backbone
        self.roi_align = partial(roi_align, **config.roi_align_config, aligned=True)
        self.features_adapters = nn.ModuleList()
        for in_channels in config.features_in_channels:
            self.features_adapters.append(ResnetBlock2D(in_channels=in_channels, out_channels=config.roi_in_channels, temb_channels=None))
        self.roi_transformers = nn.Sequential(*[
            BasicTransformerBlock(dim=config.roi_in_channels, num_attention_heads=config.heads,
                                  attention_head_dim=config.roi_in_channels // config.heads, activation_fn='gelu') for _ in
            range(config.num_transformer_blocks)])
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.output_size * config.output_size + 1, config.roi_in_channels) / config.roi_in_channels ** 0.5
        )
        self.linear_prob = nn.Sequential(nn.LayerNorm(config.roi_in_channels), nn.SiLU(), nn.Linear(config.roi_in_channels, 1))

    def forward(self, x, timestep, *, conditioning={}, entity_boxes, hw_ori=None):
        gan_model_output = self.backbone(x, timestep, return_up_features=self.config.use_up_features, **conditioning)
        roi_features = []
        for i_feature, feature in enumerate(gan_model_output.features):
            feature = self.features_adapters[i_feature](feature, temb=None)
            feature_hw = feature.shape[2:]
            scale_h, scale_w = feature_hw[1] / hw_ori[1], feature_hw[1] / hw_ori[1]
            entity_boxes_feature = entity_boxes.clone()
            entity_boxes_feature[:, 1:] = resize_coordinate(entity_boxes_feature[:, 1:], scale_h=scale_h, scale_w=scale_w)
            roi_features.append(self.roi_align(feature, entity_boxes_feature))
        if self.config.pool_method == 'mean':
            roi_features = torch.stack(roi_features, dim=0).mean(dim=0)
        else:
            raise NotImplementedError

        roi_features = rearrange(roi_features, 'b c h w -> b (h w) c')
        roi_features_global = roi_features.mean(dim=1, keepdim=True)
        roi_features = torch.cat([roi_features_global, roi_features], dim=1)
        roi_features = roi_features + self.position_embedding
        roi_feature = self.roi_transformers(roi_features)
        prob = self.linear_prob(roi_feature[:, 0])
        output = EntityGANOutput(prob=prob, roi_feature_last=roi_feature)
        return output


if __name__ == '__main__':
    model = EntityGAN().cuda()
    input = torch.randn(1, 4, 64, 64).cuda()
    entity_boxes = torch.zeros(2, 5).cuda()
    entity_boxes[0, 3] = 2
    entity_boxes[0, 4] = 2
    encoder_hidden_states = torch.randn(1, 77, 768).cuda() # Example shape for encoder hidden states
    output = model(input, timestep=1, entity_boxes=entity_boxes, conditioning={'encoder_hidden_states': encoder_hidden_states})
