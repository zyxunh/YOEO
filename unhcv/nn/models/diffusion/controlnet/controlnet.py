from diffusers.models.attention import BasicTransformerBlock
from functools import partial

from diffusers.models.controlnet import ControlNetOutput
from typing import Union, Optional, Dict, Any, Tuple

import torch
from diffusers import ControlNetModel, UNet2DConditionModel, Transformer2DModel
from torch import nn

from .transformer import BasicTransformerBlockControl
from unhcv.nn.utils import clone_conv, ReplaceModule

__all__ = ['CustomControlNetModel', 'CustomUNet2DConditionModel', 'Transformer2DModelControl']


class CustomControlNetModel(ControlNetModel):
    denoiser: UNet2DConditionModel

    def __init__(self, *args, **kwargs):
        super(CustomControlNetModel, self).__init__(*args, **kwargs)
        self.controlnet_cond_embedding = clone_conv(self.conv_in, resume_weight=False, zero_init=True)

    def custom_init(self, denoiser, collector):
        self.denoiser = denoiser
        self.collector = collector
        denoiser.requires_grad_(False)
        collector['control_channels'] = [1280] * 3 + [640] * 3 + [320] * 4
        ReplaceModule().walk_all_children(denoiser.mid_block, Transformer2DModel,
                                          replace_func=Transformer2DModelControl(collector).wrap)
        ReplaceModule().walk_all_children(denoiser.up_blocks, Transformer2DModel,
                                          replace_func=Transformer2DModelControl(collector).wrap)
        ReplaceModule().walk_all_children(denoiser.mid_block, BasicTransformerBlock,
                                          replace_func=BasicTransformerBlockControl(collector).wrap)
        ReplaceModule().walk_all_children(denoiser.up_blocks, BasicTransformerBlock,
                                          replace_func=BasicTransformerBlockControl(collector).wrap)
        self.init_for_control(denoiser, collector=collector)

    def overall_forward(self, sample, timestep, encoder_hidden_states, *, controlnet_cond, clip_embed):
        out = self(sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states, controlnet_cond=controlnet_cond)
        block_res_samples = out.down_block_res_samples + [out.mid_block_res_sample, ]
        self.collector["block_res_samples"] = block_res_samples
        self.collector["clip_embed"] = clip_embed
        noise_pred = self.denoiser(sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
        self.collector.clear()
        return noise_pred

    @staticmethod
    def init_for_control(denoiser, collector):
        denoiser.mid_block.resnets[0].forward = custom_control_forward(denoiser.mid_block.resnets[0].forward, collector=collector)
        for block in denoiser.up_blocks:
            for var in block.resnets:
                if hasattr(block, "attentions"):
                    assert len(block.attentions) == len(block.resnets)
                var.forward = custom_control_forward(var.forward, collector=collector)


class ControlOperatorBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        return self.block(*args, **kwargs)


def custom_control_forward(func, collector, name=None):
    def wrapper(*args, **kwargs):
        collector['block_res_sample'] = collector['block_res_samples'].pop()
        return func(*args, **kwargs)
    return wrapper


class CustomUNet2DConditionModel(UNet2DConditionModel):
    def init_for_control(self, collector):
        self.mid_block.resnets[0].forward = custom_control_forward(self.mid_block.resnets[0].forward, collector=collector)
        for block in self.up_blocks:
            for var in block.resnets:
                if hasattr(block, "attentions"):
                    assert len(block.attentions) == len(block.resnets)
                var.forward = custom_control_forward(var.forward, collector=collector)


class Transformer2DModelControl:
    def __init__(self, collector):
        self.collector = collector

    @staticmethod
    def forward(self: Transformer2DModel, *args, **kwargs):
        block_res_sample = self.collector["block_res_sample"]
        if not self.is_input_continuous:
            raise NotImplementedError
        self.collector["block_res_sample_proj"], _ = Transformer2DModelControl._operate_on_continuous_inputs(self, block_res_sample)
        return self._forward(*args, **kwargs)

    @staticmethod
    def _operate_on_continuous_inputs(self: Transformer2DModel, hidden_states):
        batch, _, height, width = hidden_states.shape
        hidden_states = self.norm_control(hidden_states)

        if not self.use_linear_projection:
            hidden_states = self.proj_in_control(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in_control(hidden_states)

        return hidden_states, inner_dim

    def wrap(self, module, collector=None):
        collector = self.collector
        control_channel = collector["control_channels"].pop(0)
        if module.in_channels != control_channel:
            proj_in_before_norm = torch.nn.Conv2d(control_channel, module.in_channels, kernel_size=1, stride=1, padding=0)
            module.norm_control = torch.nn.Sequential(proj_in_before_norm,
                                                      torch.nn.GroupNorm(num_groups=module.config.norm_num_groups,
                                                                         num_channels=module.in_channels, eps=1e-6,
                                                                         affine=True))
        else:
            module.norm_control = torch.nn.GroupNorm(num_groups=module.config.norm_num_groups, num_channels=module.in_channels,
                                             eps=1e-6, affine=True)
        if module.use_linear_projection:
            module.proj_in_control = torch.nn.Linear(module.in_channels, module.inner_dim)
        else:
            module.proj_in_control = torch.nn.Conv2d(module.in_channels, module.inner_dim, kernel_size=1, stride=1, padding=0)
        module._forward = module.forward
        module.forward = partial(self.forward, module)
        module.collector = collector
        return module


if __name__ == '__main__':
    pass
