from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union

import torch
from diffusers.configuration_utils import register_to_config
from torch import nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.upsampling import Upsample2D
from diffusers.utils import BaseOutput

from unhcv.common.utils import get_logger

logger = get_logger(__name__)


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


@dataclass
class BackboneOutput(BaseOutput):
    backbone_features: List[torch.Tensor] = None


@dataclass
class Unet2DBackboneOutput(BackboneOutput):
    up_block_features: List[torch.Tensor] = None


class BackboneMixin(metaclass=ABCMeta):

    @property
    @abstractmethod
    def channels(self) -> Tuple[int]:
        pass

    @property
    @abstractmethod
    def strides(self) -> Tuple[int]:
        pass


@dataclass
class Unet2DInput:
    sample: torch.Tensor = None
    timestep: torch.Tensor = None
    encoder_hidden_states: torch.Tensor = None


class Unet2DBackbone(UNet2DConditionModel, BackboneMixin):

    @register_to_config
    def __init__(self, *, sample_size: Optional[Union[int, Tuple[int, int]]] = None, in_channels: int = 4,
                 out_channels: int = 4, center_input_sample: bool = False, flip_sin_to_cos: bool = True,
                 freq_shift: int = 0, down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
            ), mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn", up_block_types: Tuple[str] = (
                    "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                 only_cross_attention: Union[bool, Tuple[bool]] = False,
                 block_out_channels: Tuple[int] = (320, 640, 1280, 1280), layers_per_block: Union[int, Tuple[int]] = 2,
                 downsample_padding: int = 1, mid_block_scale_factor: float = 1, dropout: float = 0.0,
                 act_fn: str = "silu", norm_num_groups: Optional[int] = 32, norm_eps: float = 1e-5,
                 cross_attention_dim: Union[int, Tuple[int]] = 1280,
                 transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
                 reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
                 encoder_hid_dim: Optional[int] = None, encoder_hid_dim_type: Optional[str] = None,
                 attention_head_dim: Union[int, Tuple[int]] = 8,
                 num_attention_heads: Optional[Union[int, Tuple[int]]] = None, dual_cross_attention: bool = False,
                 use_linear_projection: bool = False, class_embed_type: Optional[str] = None,
                 addition_embed_type: Optional[str] = None, addition_time_embed_dim: Optional[int] = None,
                 num_class_embeds: Optional[int] = None, upcast_attention: bool = False,
                 resnet_time_scale_shift: str = "default", resnet_skip_time_act: bool = False,
                 resnet_out_scale_factor: float = 1.0, time_embedding_type: str = "positional",
                 time_embedding_dim: Optional[int] = None, time_embedding_act_fn: Optional[str] = None,
                 timestep_post_act: Optional[str] = None, time_cond_proj_dim: Optional[int] = None,
                 conv_in_kernel: int = 3, conv_out_kernel: int = 3,
                 projection_class_embeddings_input_dim: Optional[int] = None, attention_type: str = "default",
                 class_embeddings_concat: bool = False, mid_block_only_cross_attention: Optional[bool] = None,
                 cross_attention_norm: Optional[str] = None, addition_embed_type_num_heads: int = 64):
        super().__init__(sample_size=sample_size, in_channels=in_channels, out_channels=out_channels,
                         center_input_sample=center_input_sample, flip_sin_to_cos=flip_sin_to_cos,
                         freq_shift=freq_shift, down_block_types=down_block_types, mid_block_type=mid_block_type,
                         up_block_types=up_block_types, only_cross_attention=only_cross_attention,
                         block_out_channels=block_out_channels, layers_per_block=layers_per_block,
                         downsample_padding=downsample_padding, mid_block_scale_factor=mid_block_scale_factor,
                         dropout=dropout, act_fn=act_fn, norm_num_groups=norm_num_groups, norm_eps=norm_eps,
                         cross_attention_dim=cross_attention_dim,
                         transformer_layers_per_block=transformer_layers_per_block,
                         reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
                         encoder_hid_dim=encoder_hid_dim, encoder_hid_dim_type=encoder_hid_dim_type,
                         attention_head_dim=attention_head_dim, num_attention_heads=num_attention_heads,
                         dual_cross_attention=dual_cross_attention, use_linear_projection=use_linear_projection,
                         class_embed_type=class_embed_type, addition_embed_type=addition_embed_type,
                         addition_time_embed_dim=addition_time_embed_dim, num_class_embeds=num_class_embeds,
                         upcast_attention=upcast_attention, resnet_time_scale_shift=resnet_time_scale_shift,
                         resnet_skip_time_act=resnet_skip_time_act, resnet_out_scale_factor=resnet_out_scale_factor,
                         time_embedding_type=time_embedding_type, time_embedding_dim=time_embedding_dim,
                         time_embedding_act_fn=time_embedding_act_fn, timestep_post_act=timestep_post_act,
                         time_cond_proj_dim=time_cond_proj_dim, conv_in_kernel=conv_in_kernel,
                         conv_out_kernel=conv_out_kernel,
                         projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
                         attention_type=attention_type, class_embeddings_concat=class_embeddings_concat,
                         mid_block_only_cross_attention=mid_block_only_cross_attention,
                         cross_attention_norm=cross_attention_norm,
                         addition_embed_type_num_heads=addition_embed_type_num_heads)
        self.upsample = Upsample2D(block_out_channels[0], use_conv=True, out_channels=block_out_channels[0])
        self.block_out_channels = block_out_channels

    def forward(self, *args, **kwargs) -> BackboneOutput:
        if isinstance(args[0], Unet2DInput):
            input = args[0]
            kwargs['sample'] = input.sample
            kwargs['timestep'] = input.timestep
            kwargs['encoder_hidden_states'] = input.encoder_hidden_states
            args = ()
        up_block_features = super().forward(*args, **kwargs).up_block_features
        up_block_features = (*up_block_features[:-1], self.upsample(up_block_features[-1]))
        out = BackboneOutput(backbone_features=up_block_features[::-1])
        return out

    @property
    def channels(self) -> Tuple[int]:
        return self.block_out_channels

    @property
    def strides(self) -> Tuple[int]:
        strides = [4, 8, 32, 64]
        return strides


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

if __name__ == '__main__':
    from unhcv.common.utils import obj_load, find_path

    config = obj_load(find_path("model/stable-diffusion-v1-5/unet/config.json"))
    config.update(dict(out_channels=None))
    # model = UNet2DConditionModel.from_config(config, return_unused_kwargs=True)
    # breakpoint()
    model = Unet2DBackbone.from_config(config)
    model = model.cuda()

    sample = torch.zeros([1, 4, 64, 64], dtype=torch.float32).cuda()
    timestep = torch.zeros([1], dtype=torch.int64).cuda()
    encoder_hidden_states = torch.zeros([1, 77, 768], dtype=torch.float32).cuda()

    out = model(sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
    import peft
    breakpoint()
    pass