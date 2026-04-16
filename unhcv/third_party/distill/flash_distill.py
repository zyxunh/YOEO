import dataclasses
import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Union, Literal, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, LCMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, deprecate, unscale_lora_layers
from lpips import lpips
from torch import nn
from tqdm import tqdm
from unhcv.common import CfgNode
from unhcv.common.image import mask2bounding_box, masks2bboxes, visual_tensor
from unhcv.common.types import default_factory, ModelOutput
from unhcv.nn.loss.lpips import LPIPSUnh
from unhcv.nn.models.diffusion import ODE
# from unhcv.nn.models.diffusion import VAE
from unhcv.nn.utils import ReplaceModule, load_checkpoint
from unhcv.common.utils import find_path

__all__ = ["FlashDistill", "FlashDistillConfig", "Discriminator"]

from unhcv.third_party.distill.gan_model import EntityGAN

from unhcv.third_party.remove import MARSLoss


def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Extracts values from a tensor into a new tensor using indices from another tensor.

    :param a: the tensor to extract values from.
    :param t: the tensor containing the indices.
    :param x_shape: the shape of the tensor to extract values into.
    :return: a new tensor containing the extracted values.
    """

    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def gaussian_mixture(k, locs, var, mode_probs=None):
    if mode_probs is None:
        mode_probs = [1 / len(locs)] * len(locs)

    def _gaussian(x):
        prob = [
            mode_probs[i] * torch.exp(-torch.tensor([(x - loc) ** 2 / var]))
            for i, loc in enumerate(locs)
        ]
        # prob.append(mode_prob * torch.exp(-torch.tensor([(x) ** 2 / var])))
        return sum(prob)

    return _gaussian


@dataclasses.dataclass
class FLashUNet2DConditionOutput(ModelOutput):
    sample: torch.Tensor = None
    up_block_res_samples: torch.Tensor = None


class FLashUNet2DConditionModel:
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
        return_res_samples: bool = False,
    ) -> Union[FLashUNet2DConditionOutput, Tuple]:
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

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        upsample_block_extra_kwargs = {}
        up_block_res_samples_ = ()
        if return_res_samples:
            upsample_block_extra_kwargs['return_res_samples'] = True
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
                    **upsample_block_extra_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    **upsample_block_extra_kwargs,
                )
            if return_res_samples:
                sample, up_block_res_samples = sample
                up_block_res_samples_ += up_block_res_samples

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

        return FLashUNet2DConditionOutput(sample=sample, up_block_res_samples=up_block_res_samples_)


@dataclass
class FlashDistillConfig(CfgNode):
    """
    Configuration class for the FlashDiffusion model.

    Args:

        K (List[int]): The list of number of timesteps for each stage. Defaults to [32, 32, 32, 32, 32].
        num_iterations_per_K (List[int]): The number of iterations for each stage. Defaults to [5000, 10000, 15000, 20000, 25000].
        guidance_scale_min (List[float]): The minimum guidance scale for each stage. Defaults to 3.0.
        guidance_scale_max (List[float]): The maximum guidance scale for each stage. Defaults to 7.0.
        distill_loss_type (Literal["l2", "l1", "lpips"]): The type of distillation loss to use. Defaults to "l2". Choices are "l2", "l1", "lpips".
        ucg_keys (List[str]): The keys to use for classifier-guidance with the teacher model. Defaults to ["text"].
        timestep_distribution (Literal["gaussian", "uniform", "mixture"]): The distribution of timesteps to use. Defaults to "mixture". Choices are "gaussian", "uniform", "mixture".
        mixture_num_components (List[int]): The number of components in the timestep mixture distribution for each stage. Defaults to 4.
        mixture_var (List[float]): The variance of the timestep mixture distribution for each stage. Defaults to 0.5.
        adapter_conditioning_scale (float): The scale of the adapter conditioning. Defaults to 1.0.
        adapter_input_key (Optional[str]): The key to use for the adapter input. Defaults to None.
        use_dmd_loss (bool): Whether to use the DMD loss. Defaults to False.
        dmd_loss_scale (Union[float, List[float]]): The scale of the DMD loss for each stage. Defaults to 1.0.
        distill_loss_scale (Union[float, List[float]]): The scale of the distillation loss for each stage. Defaults to 1.0.
        adversarial_loss_scale (Union[float, List[float]]): The scale of the adversarial loss for each stage. Defaults to 1.0.
        gan_loss_type (Literal["hinge", "vanilla", "non-saturating", "wgan", "lsgan"]): The type of GAN loss to use. Defaults to "hinge". Choices are "hinge", "vanilla", "non-saturating", "wgan", "lsgan".
        mode_probs (Optional[List[List[float]]]): The mode probabilities for the timestep mixture distribution. Defaults to None.
        use_teacher_as_real (bool): Whether to use the teacher model as the real image. Defaults to False.
        use_empty_prompt (bool): Whether to use an empty prompt for classifier-free guidance (useful for SD1.5 and Pixart-Alpha). Defaults to False.
    """

    K: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    num_iterations_per_K: List[int] = field(
        default_factory=lambda: [5000, 5000, 5000, 5000]
    )
    guidance_scale_min: Union[float, List[float]] = 1.0
    guidance_scale_max: Union[float, List[float]] = 1.0
    distill_loss_type: Literal['l2', "l1", "lpips", "l2_rgb"] = "lpips"
    ucg_keys: List[str] = field(default_factory=lambda: ["text"])
    timestep_distribution: Literal["gaussian", "uniform", "mixture"] = "mixture"
    mixture_num_components: Union[int, List[int]] = 4
    mixture_var: Union[float, List[float]] = 0.5
    adapter_conditioning_scale: float = 1.0
    adapter_input_key: Optional[str] = None
    use_dmd_loss: bool = True
    dmd_loss_scale: Union[float, List[float]] = (0, 0.3, 0.5, 0.7)
    distill_loss_scale: Union[float, List[float]] = 1.0
    adversarial_loss_scale: Union[float, List[float]] = (0, 0.1, 0.2, 0.3)
    gan_loss_type: Literal["hinge", "vanilla", "non-saturating", "wgan", "lsgan"] = (
        "lsgan"
    )
    mode_probs: Optional[List[List[float]]] = default_factory([
                                                    [0.0, 0.0, 0.5, 0.5],
                                                    [0.1, 0.3, 0.3, 0.3],
                                                    [0.25, 0.25, 0.25, 0.25],
                                                    [0.4, 0.2, 0.2, 0.2],])
    use_teacher_as_real: bool = False
    use_empty_prompt: bool = True

    #
    train_dtype: type = torch.bfloat16
    return_rgb_output: bool = False

    discriminator_pretrained_model: str = None
    use_mars: bool = False
    train_D_on_non_pair_label: bool = True
    train_D_only_on_non_pair_label_fake: bool = False
    train_ode_mode: bool = False
    # GAN
    gan_model: Literal["default", "EntityGAN"] = "default"
    entity_ios_thres: float = 0.05
    entity_score_thres: float = 0.2
    max_entity_bboxes_num: int = 200
    #
    use_mars_predictor: bool = None
    use_student_timestep_in_dmd: bool = False

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.mixture_num_components, int):
            self.mixture_num_components = [self.mixture_num_components] * len(self.K)

        if isinstance(self.guidance_scale_min, float):
            self.guidance_scale_min = [self.guidance_scale_min] * len(self.K)

        if isinstance(self.guidance_scale_max, float):
            self.guidance_scale_max = [self.guidance_scale_max] * len(self.K)

        if isinstance(self.mixture_num_components, int):
            self.mixture_num_components = [self.mixture_num_components] * len(self.K)

        if isinstance(self.mixture_var, float):
            self.mixture_var = [self.mixture_var] * len(self.K)

        if isinstance(self.distill_loss_scale, float):
            self.distill_loss_scale = [self.distill_loss_scale] * len(self.K)

        if isinstance(self.dmd_loss_scale, float):
            self.dmd_loss_scale = [self.dmd_loss_scale] * len(self.K)

        if isinstance(self.adversarial_loss_scale, float):
            self.adversarial_loss_scale = [self.adversarial_loss_scale] * len(self.K)

        if self.mode_probs is None:
            self.mode_probs = [
                [1 / mixtures] * mixtures for mixtures in self.mixture_num_components
            ]

        for i in range(len(self.K)):
            assert len(self.mode_probs[i]) == self.mixture_num_components[i], (
                f"Number of mode probabilities must match number of mixture components for stage {i}, "
                f"got {len(self.mode_probs[i])} mode probabilities and {self.mixture_num_components[i]} mixture components"
            )

        assert len(self.K) == len(
            self.num_iterations_per_K
        ), f"Number of timesteps must match number of iterations, got {len(self.K)} timesteps and {len(self.num_iterations_per_K)} iterations"

        assert len(self.K) == len(
            self.mode_probs
        ), f"Number of timesteps must match number of mode probabilities, got {len(self.K)} timesteps and {len(self.mode_probs)} mode probabilities"

        self.discriminator_pretrained_model = find_path(self.discriminator_pretrained_model)



class FlashDistill(nn.Module):
    mars_loss: MARSLoss

    def __init__(self, config: FlashDistillConfig, student_denoiser, teacher_denoiser=None,
                 teacher_noise_scheduler: DDPMScheduler = None, teacher_sampling_noise_scheduler: DDPMScheduler = None,
                 sampling_noise_scheduler: LCMScheduler = None, vae=None, discriminator: nn.Module = None,
                 #
                 collector=None, custom_forward=None):
        super().__init__()
        self.config = config = FlashDistillConfig.from_other(config)
        self.student_denoiser = student_denoiser
        if teacher_denoiser is None:
            teacher_denoiser = deepcopy(student_denoiser)
        self.teacher_denoiser = teacher_denoiser
        self.teacher_noise_scheduler = teacher_noise_scheduler
        self.teacher_sampling_noise_scheduler = teacher_sampling_noise_scheduler
        self.guidance_scale_min = config.guidance_scale_min
        self.guidance_scale_max = config.guidance_scale_max
        self.K = config.K
        self.num_iterations_per_K = config.num_iterations_per_K
        self.distill_loss_type = config.distill_loss_type
        self.timestep_distribution = config.timestep_distribution
        self.iter_steps = 0
        self.mixture_num_components = config.mixture_num_components
        self.mixture_var = config.mixture_var
        self.adapter_conditioning_scale = config.adapter_conditioning_scale
        self.use_dmd_loss = config.use_dmd_loss
        self.dmd_loss_scale = config.dmd_loss_scale
        self.distill_loss_scale = config.distill_loss_scale
        self.discriminator = discriminator
        self.adversarial_loss_scale = config.adversarial_loss_scale
        self.gan_loss_type = config.gan_loss_type
        self.mode_probs = config.mode_probs
        self.use_teacher_as_real = config.use_teacher_as_real
        self.use_empty_prompt = config.use_empty_prompt
        self.adapter = None

        self.disc_update_counter = 0

        if self.discriminator is None:
            logging.warning(
                "No discriminator provided. Adversarial loss will be ignored."
            )
            self.use_adversarial_loss = False

        if config.gan_model == "default":
            self.disc_backbone = self.teacher_denoiser
        elif config.gan_model == "EntityGAN":
            del self.discriminator
            self.discriminator = EntityGAN(backbone=self.teacher_denoiser)

        if self.distill_loss_type == "lpips":
            self.lpips = LPIPSUnh.init()
            # self.lpips = lpips.LPIPS(net="vgg")
        self.vae = vae

        self.sampling_noise_scheduler = sampling_noise_scheduler

        self.K_steps = np.cumsum(self.num_iterations_per_K)
        self.K_prev = self.K[0]

        if teacher_noise_scheduler is not None:
            if hasattr(teacher_noise_scheduler, "alphas_cumprod"):
                self.register_buffer(
                    "sqrt_alpha_cumprod",
                    torch.sqrt(self.teacher_noise_scheduler.alphas_cumprod),
                )
                self.register_buffer(
                    "sigmas",
                    torch.sqrt(1 - self.teacher_noise_scheduler.alphas_cumprod),
                )
            elif hasattr(teacher_noise_scheduler, "sigmas"):
                self.register_buffer(
                    "sqrt_alpha_cumprod",
                    torch.sqrt(1 - teacher_noise_scheduler.sigmas**2),
                )
                self.register_buffer("sigmas", teacher_noise_scheduler.sigmas)

        ####
        self.collector = collector
        teacher_denoiser.collector = collector
        if config.gan_model == "default":
            ReplaceModule.setattr_function(teacher_denoiser, "_forward", FLashUNet2DConditionModel.forward)
        elif config.gan_model == "EntityGAN":
            ReplaceModule.setattr_function(teacher_denoiser, "_forward", teacher_denoiser.forward)
        ReplaceModule.setattr_function(teacher_denoiser, "forward", custom_forward)
        student_denoiser.collector = collector
        ReplaceModule.setattr_function(student_denoiser, "_forward", FLashUNet2DConditionModel.forward)
        ReplaceModule.setattr_function(student_denoiser, "forward", custom_forward)
        self.teacher_denoiser = teacher_denoiser.to(self.config.train_dtype)
        if hasattr(self, 'lpips'):
            self.lpips = self.lpips.to(self.config.train_dtype)

        if self.config.discriminator_pretrained_model is not None:
            load_checkpoint(self.discriminator, self.config.discriminator_pretrained_model)

        self.custom_lcm_scheduler: LCMScheduler = None

    def _get_timesteps(
        self, num_samples: int = 1, K: int = 1, K_step: int = 1, device="cpu"
    ):
        # Get the timesteps for the current K
        self.teacher_noise_scheduler.set_timesteps(K)

        if self.timestep_distribution == "uniform":
            prob = torch.ones(K) / K
        elif self.timestep_distribution == "gaussian":
            prob = [torch.exp(-torch.tensor([(i - K / 2) ** 2 / K])) for i in range(K)]
            prob = torch.tensor(prob) / torch.sum(torch.tensor(prob))
        elif self.timestep_distribution == "mixture":
            mixture_num_components = self.mixture_num_components[K_step]
            mode_probs = self.mode_probs[K_step]

            # Define targeted timesteps
            locs = [
                i * (K // mixture_num_components)
                for i in range(0, mixture_num_components)
            ]
            mixture_var = self.mixture_var[K_step]
            prob = [
                gaussian_mixture(
                    K,
                    locs=locs,
                    var=mixture_var,
                    mode_probs=mode_probs,
                )(i)
                for i in range(K)
            ]
            prob = torch.tensor(prob) / torch.sum(torch.tensor(prob))

        start_idx = torch.multinomial(prob, 1)

        # start_idx = torch.randint(0, len(self.teacher_noise_scheduler.timesteps), (1,))

        start_timestep = (
            self.teacher_noise_scheduler.timesteps[start_idx]
            .to(device)
            .repeat(num_samples)
        )

        return start_idx, start_timestep

    def forward(self, z, *args, student_conditioning=None, conditioning, unconditional_conditioning, batch: Dict[str, Any]=None, step=0, use_gt=True, use_mars=False, data_tag: Literal["pair_label", "non_pair_label"], **kwargs):
        loss_dict = {}
        self.iter_steps += 1

        if student_conditioning is None:
            student_conditioning = conditioning

        # Compute the T2I adapter features
        down_intrablock_additional_residuals = None

        # Get K for the current step
        if self.iter_steps > self.K_steps[-1]:
            K_step = len(self.K) - 1
        else:
            K_step = np.argmax(self.iter_steps < self.K_steps)
        K = self.K[K_step]
        guidance_min = self.guidance_scale_min[K_step]
        guidance_max = self.guidance_scale_max[K_step]
        if K != self.K_prev:
            self.K_prev = K
            if self.switch_teacher:
                print("Switching teacher")
                self.teacher_denoiser = deepcopy(self.student_denoiser)
                self.teacher_denoiser.freeze()

        # Create noisy samples
        noise = torch.randn_like(z)

        # Sample the timesteps
        self.K_tmp, self.K_step_tmp = K, K_step
        if use_gt:
            start_idx, start_timestep = self._get_timesteps(
                num_samples=z.shape[0], K=K, K_step=K_step, device=z.device
            )
        else:
            self.teacher_noise_scheduler.set_timesteps(K)
            start_idx = torch.tensor([0], dtype=torch.long)
            start_timestep = (self.teacher_noise_scheduler.timesteps[start_idx]
                              .to(noise)
                              .repeat(noise.size(0)))
        print('start_timestep', start_timestep)

        if start_idx == 0:
            noisy_sample_init = noise
            noisy_sample_init *= self.teacher_noise_scheduler.init_noise_sigma
            noisy_sample_init_student = noise

        else:
            # Add noise to sample
            noisy_sample_init = self.teacher_noise_scheduler.add_noise(
                z, noise, start_timestep
            )
            noisy_sample_init_student = noisy_sample_init
        student_output_feature = None
        student_extra_kwargs = {}
        if self.config.use_mars_predictor:
            student_extra_kwargs["return_res_samples"] = True
            student_extra_kwargs["return_up_features"] = True

        if self.config.train_ode_mode and data_tag == "non_pair_label":
            train_start_idx = random.randint(0, 3)
            # print('random train_start_idx', train_start_idx)
            if train_start_idx != 0:
                student_output = ODE.ode(self.custom_lcm_scheduler, self.student_denoiser, noisy_sample_init_student, condition=student_conditioning, generator=None, guidance_scale=1, steps=4, requires_grad=False, start_idx=0, end_idx=train_start_idx)
            else:
                student_output = noisy_sample_init_student
            _, student_output = ODE.ode(self.custom_lcm_scheduler, self.student_denoiser, student_output, condition=student_conditioning,
                                     generator=None, guidance_scale=1, steps=4, requires_grad=True, start_idx=train_start_idx, end_idx=train_start_idx+1, return_original_sample=True)

        else:
            noisy_sample_init_ = self.teacher_noise_scheduler.scale_model_input(
                noisy_sample_init_student, start_timestep
            )

            # Get student denoiser output
            student_noise_pred = self.student_denoiser(
                sample=noisy_sample_init_,
                timestep=start_timestep,
                **student_conditioning,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                **student_extra_kwargs
            )
            if self.config.use_mars_predictor:
                student_output_feature = student_noise_pred
                student_noise_pred = student_noise_pred.sample

            c_skip, c_out = self._scalings_for_boundary_conditions(start_timestep)

            c_skip = append_dims(c_skip, noisy_sample_init_student.ndim)
            c_out = append_dims(c_out, noisy_sample_init_student.ndim)

            student_output = self._predicted_x_0(
                student_noise_pred,
                start_timestep.type(torch.int64),
                noisy_sample_init_student,
                "epsilon",
                self.sqrt_alpha_cumprod,
                self.sigmas,
                z,
            )

        noisy_sample = noisy_sample_init.clone().detach()

        guidance_scale = (
            torch.rand(1).to(z.device) * (guidance_max - guidance_min) + guidance_min
        )

        with torch.no_grad():
            for t in self.teacher_noise_scheduler.timesteps[start_idx:]:
                timestep = torch.tensor([t], device=z.device).repeat(z.shape[0])

                noisy_sample_ = self.teacher_noise_scheduler.scale_model_input(
                    noisy_sample, t
                ).to(self.config.train_dtype)

                # Denoise sample
                noise_pred = cond_noise_pred = self.teacher_denoiser(
                    sample=noisy_sample_,
                    timestep=timestep,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    *args,
                    **conditioning,
                    **kwargs,
                )

                if guidance_scale > 1:
                    uncond_noise_pred = self.teacher_denoiser(
                        sample=noisy_sample_,
                        timestep=timestep,
                        down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                        *args,
                        **unconditional_conditioning,
                        **kwargs,
                    )

                    # Make CFG
                    noise_pred = (
                        guidance_scale * cond_noise_pred
                        + (1 - guidance_scale) * uncond_noise_pred
                    )

                # Make one step on the reverse diffusion process
                noisy_sample = self.teacher_noise_scheduler.step(
                    noise_pred, t, noisy_sample, return_dict=False
                )[0]

        teacher_output = noisy_sample

        if self.config.train_ode_mode and data_tag == "non_pair_label":
            pass
        else:
            student_output = c_skip * noisy_sample_init + c_out * student_output

        loss, lpips_output = (
            self._distill_loss(student_output, teacher_output, use_mars=use_mars)
        )
        loss = loss*self.distill_loss_scale[K_step]
        loss_dict['loss_distill'] = loss

        if self.use_dmd_loss:
            dmd_loss = self._dmd_loss(
                student_output.to(self.config.train_dtype),
                student_conditioning,
                conditioning,
                unconditional_conditioning,
                down_intrablock_additional_residuals,
                K,
                K_step,
                extra_output=lpips_output,
            )
            loss_dict['loss_dmd'] = dmd_loss * self.dmd_loss_scale[K_step]
            # loss += dmd_loss * self.dmd_loss_scale[K_step]

        gan_loss = self._gan_loss(
            z,
            batch,
            student_output,
            teacher_output,
            conditioning,
            down_intrablock_additional_residuals,
            step=step, extra_output=lpips_output,
            data_tag=data_tag
        )
        print("GAN loss", gan_loss)
        loss_dict['loss_gan'] = self.adversarial_loss_scale[K_step] * gan_loss[0]
        # loss += self.adversarial_loss_scale[K_step] * gan_loss[0]
        loss_disc = gan_loss[1]

        return {
            "loss": dict(loss_disc=loss_disc, **loss_dict),
            "teacher_output": teacher_output,
            "student_output": student_output,
            "noisy_sample": noisy_sample_init,
            "start_timestep": start_timestep[0].item(),
            "lpips_output": lpips_output,
            "student_output_feature": student_output_feature,
            "loss_dict": loss_dict,
        }

    def _distill_loss(self, student_output, teacher_output, use_mars=False):
        if self.config.return_rgb_output or self.distill_loss_type == "lpips":
            decoded_student = self.vae.decode(student_output.to(self.config.train_dtype)).clamp(-1, 1)
            decoded_teacher = self.vae.decode(teacher_output.to(self.config.train_dtype)).clamp(-1, 1)
            extra_output = dict(decoded_student=decoded_student, decoded_teacher=decoded_teacher)
        else:
            extra_output = {}

        if self.distill_loss_type == "l2":
            distill_loss = torch.mean(
                ((student_output - teacher_output) ** 2).reshape(
                    student_output.shape[0], -1
                ),
                1,
            ).mean()
        elif self.distill_loss_type == "l1":
            distill_loss =  torch.mean(
                torch.abs(student_output - teacher_output).reshape(
                    student_output.shape[0], -1
                ),
                1,
            ).mean()
        elif self.distill_loss_type == "lpips":
            # center crop patches of size 64x64
            # crop_h = (student_output.shape[2] - 64) // 2
            # crop_w = (student_output.shape[3] - 64) // 2
            # student_output = student_output[
            #     :, :, crop_h : crop_h + 64, crop_w : crop_w + 64
            # ]
            # teacher_output = teacher_output[
            #     :, :, crop_h : crop_h + 64, crop_w : crop_w + 64
            # ]
            # self.lpips = self.lpips.to(student_output.device)
            if self.config.use_mars and use_mars:
                mars_output = self.mars_loss.predict_mars_masks(decoded_teacher, self.collector['inpainting_mask_ori'], ios_thres=0.85)
                pred_mars_masks = F.interpolate(mars_output.pred_mars_masks[:, None], decoded_teacher.shape[2:], mode="bilinear")
                pred_mars_masks = (pred_mars_masks.sigmoid() > 0.1)
                pred_mars_masks_not = ~pred_mars_masks
                distill_loss = self.lpips(decoded_student * pred_mars_masks_not, decoded_teacher * pred_mars_masks_not).mean()
                extra_output['pred_mars_masks'] = pred_mars_masks
                extra_output['pred_masks_sigmoid'] = mars_output.pred_masks_sigmoid
                extra_output['mars_output'] = mars_output
                extra_output['pred_probs'] = mars_output.pred_probs
            else:
                distill_loss = self.lpips(decoded_student, decoded_teacher).mean()
        elif self.distill_loss_type == "l2_rgb":
            distill_loss = F.mse_loss(decoded_student, decoded_teacher, reduction="none")
            if self.config.use_mars and use_mars:
                mars_output = self.mars_loss.predict_mars_masks(decoded_teacher, self.collector['inpainting_mask_ori'])
                pred_mars_masks = F.interpolate(mars_output.pred_mars_masks[:, None], decoded_teacher.shape[2:], mode="bilinear")
                pred_mars_masks = (pred_mars_masks.sigmoid() > 0.1)
                distill_loss = distill_loss * ~pred_mars_masks
                extra_output['pred_mars_masks'] = pred_mars_masks
            distill_loss_mask = distill_loss.detach()
            distill_loss = distill_loss.mean() * 100
            extra_output['distill_loss_mask'] = distill_loss_mask
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        return distill_loss, extra_output

    def _dmd_loss(
        self,
        student_output,
        student_conditioning,
        conditioning,
        unconditional_conditioning,
        down_intrablock_additional_residuals,
        K,
        K_step,
        extra_output,
    ):
        """
        Compute the DMD loss
        """

        # Sample noise
        noise = torch.randn_like(student_output)

        if self.config.use_student_timestep_in_dmd:
            random_indices = torch.multinomial(torch.tensor([1, 1, 1.], dtype=torch.float), num_samples=noise.shape[0], replacement=True)
            timestep = torch.tensor([749, 499, 249], dtype=torch.long, device='cuda')[random_indices]
            print('timestep dmd', timestep)
        else:
            timestep = torch.randint(
                0,
                self.teacher_noise_scheduler.config.num_train_timesteps,
                (student_output.shape[0],),
                device=student_output.device,
            )

        # Create noisy sample
        noisy_student = self.teacher_noise_scheduler.add_noise(
            student_output, noise, timestep
        )

        with torch.no_grad():

            cond_real_noise_pred = self.teacher_denoiser(
                sample=noisy_student,
                timestep=timestep,
                **conditioning,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            )

            cond_fake_noise_pred = self.student_denoiser(
                sample=noisy_student,
                timestep=timestep,
                **student_conditioning,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            )

            guidance_scale = (
                torch.rand(1).to(student_output.device)
                * (self.guidance_scale_max[K_step] - self.guidance_scale_min[K_step])
                + self.guidance_scale_min[K_step]
            )

            if guidance_scale > 1:
                uncond_real_noise_pred = self.teacher_denoiser(
                    sample=noisy_student,
                    timestep=timestep,
                    **unconditional_conditioning,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                )

        if guidance_scale > 1:
            real_noise_pred = (
                guidance_scale * cond_real_noise_pred
                + (1 - guidance_scale) * uncond_real_noise_pred
            )
        else:
            real_noise_pred = cond_real_noise_pred

        fake_noise_pred = cond_fake_noise_pred

        score_real = -real_noise_pred
        score_fake = -fake_noise_pred

        alpha_prod_t = self.teacher_noise_scheduler.alphas_cumprod.to(
            device=student_output.device, dtype=student_output.dtype
        )[timestep]
        beta_prod_t = 1.0 - alpha_prod_t

        coeff = (
            (score_fake - score_real)
            * beta_prod_t.view(-1, 1, 1, 1) ** 0.5
            / alpha_prod_t.view(-1, 1, 1, 1) ** 0.5
        )

        student_output = student_output.float()

        pred_x_0_student = self._predicted_x_0(
            real_noise_pred.float(),
            timestep,
            noisy_student.float(),
            "epsilon",
            self.sqrt_alpha_cumprod,
            self.sigmas,
            student_output,
        )

        if self.config.use_mars:
            pred_mars_masks = extra_output.get('pred_mars_masks', None)
            if pred_mars_masks is not None:
                pred_mars_masks_not = 1 - F.adaptive_avg_pool2d(pred_mars_masks.to(student_output), output_size=student_output.shape[2:])
                # student_output = student_output * pred_mars_masks_not
                # pred_x_0_student = pred_x_0_student * pred_mars_masks_not
                coeff = coeff * pred_mars_masks_not

        weight = (
            1.0
            / (
                (student_output - pred_x_0_student).abs().mean([1, 2, 3], keepdim=True)
                + 1e-5
            ).detach()
        )
        dmd_target = (student_output - weight * coeff).detach()
        extra_output["dmd_target"] = dmd_target
        return F.mse_loss(
            student_output, dmd_target, reduction="mean"
        )

    def _gan_loss(
        self,
        z,
        batch,
        student_output,
        teacher_output,
        conditioning,
        down_intrablock_additional_residuals=None,
        step=0, extra_output=None, data_tag=None
    ):
        if step % 2 == 0:
            training_G = True
        else:
            training_G = False
        if not training_G:
            student_output = student_output.detach()
            teacher_output = teacher_output.detach()

        self.disc_update_counter += 1

        # Sample noise
        noise = torch.randn_like(student_output)

        if self.use_teacher_as_real:
            real = teacher_output

        # TODO: real is not real
        else:
            real = z

        # Selected timesteps
        selected_timesteps = [10, 250, 500, 750]
        prob = torch.tensor([0.25, 0.25, 0.25, 0.25])

        # Sample the timesteps
        idx = prob.multinomial(student_output.shape[0], replacement=True).to(
            student_output.device
        )
        timesteps = torch.tensor(
            selected_timesteps, device=student_output.device, dtype=torch.long
        )[idx]

        # Create noisy sample
        noisy_fake = self.teacher_noise_scheduler.add_noise(
            student_output, noise, timesteps
        )
        noisy_real = self.teacher_noise_scheduler.add_noise(real, noise, timesteps)

        # Concatenate noisy samples
        noisy_sample = torch.cat([noisy_fake, noisy_real], dim=0)

        # Concatenate conditionings
        if conditioning is not None:
            conditioning = {
                    k: torch.cat([v, v], dim=0) for k, v in conditioning.items()
            }
            for k, v in self.collector.items():
                if k != "inpainting_mask_ori":
                    self.collector[k] = torch.cat([v, v], dim=0)

        # Concatenate timesteps
        timestep = torch.cat([timesteps, timesteps], dim=0)

        if self.adapter:
            for k, v in enumerate(down_intrablock_additional_residuals):
                down_intrablock_additional_residuals[k] = torch.cat([v, v], dim=0)

        else:
            down_intrablock_additional_residuals = None

        if self.config.gan_model == "default":
            # Predict noise level using denoiser
            denoised_sample = self.disc_backbone(
                sample=noisy_sample.to(self.config.train_dtype),
                timestep=timestep,
                **conditioning,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                return_intermediate=True,
            )

            denoised_sample_fake, denoised_sample_real = denoised_sample.chunk(2, dim=0)

            if self.gan_loss_type == "wgan":
                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
                if step % 2 == 0:
                    loss_G = -self.discriminator(denoised_sample_fake).mean()
                    loss_D = 0
                else:
                    loss_D = (
                        -self.discriminator(denoised_sample_real).mean()
                        + self.discriminator(denoised_sample_fake.detach()).mean()
                    )
                    loss_G = 0

            elif self.gan_loss_type == "lsgan":
                valid = torch.ones(student_output.size(0), 1, device=noise.device)
                fake = torch.zeros(noise.size(0), 1, device=noise.device)
                if step % 2 == 0:
                    loss_G = F.mse_loss(
                        torch.sigmoid(self.discriminator(denoised_sample_fake)), valid
                    )
                    loss_D = 0
                else:
                    if self.config.train_D_only_on_non_pair_label_fake and data_tag == "non_pair_label":
                        real_weight = 0
                    else:
                        real_weight = 1
                    loss_D = 0.5 * (
                        F.mse_loss(
                            torch.sigmoid(self.discriminator(denoised_sample_real)), valid
                        ) * real_weight
                        + F.mse_loss(
                            torch.sigmoid(
                                self.discriminator(denoised_sample_fake.detach())
                            ),
                            fake,
                        )
                    )
                    loss_G = 0
            elif self.gan_loss_type == "hinge":
                if step % 2 == 0:
                    loss_G = -self.discriminator(denoised_sample_fake).mean()
                    loss_D = 0
                else:
                    loss_D = (
                        F.relu(1.0 - self.discriminator(denoised_sample_real)).mean()
                        + F.relu(
                            1.0 + self.discriminator(denoised_sample_fake.detach())
                        ).mean()
                    )
                    loss_G = 0

            elif self.gan_loss_type == "non-saturating":
                if step % 2 == 0:
                    loss_G = -torch.mean(
                        torch.log(
                            torch.sigmoid(self.discriminator(denoised_sample_fake)) + 1e-8
                        )
                    )
                    loss_D = 0

                else:
                    loss_D = -torch.mean(
                        torch.log(
                            torch.sigmoid(self.discriminator(denoised_sample_real)) + 1e-8
                        )
                        + torch.log(
                            1
                            - torch.sigmoid(
                                self.discriminator(denoised_sample_fake.detach())
                            )
                            + 1e-8
                        )
                    )
                    loss_G = 0
            else:
                if step % 2 == 0:
                    valid = torch.ones(student_output.size(0), 1, device=noise.device)
                    loss_G = F.binary_cross_entropy_with_logits(
                        self.discriminator(denoised_sample_fake), valid
                    )
                    loss_D = 0

                else:
                    valid = torch.ones(student_output.size(0), 1, device=noise.device)
                    real = F.binary_cross_entropy_with_logits(
                        self.discriminator(denoised_sample_real), valid
                    )
                    fake = torch.zeros(noise.size(0), 1, device=noise.device)
                    fake = F.binary_cross_entropy_with_logits(
                        self.discriminator(denoised_sample_fake.detach()), fake
                    )
                    loss_D = real + fake
                    loss_G = 0
        elif self.config.gan_model == "EntityGAN":
            get_pos_indices = lambda x: (x.pred_probs > self.config.entity_score_thres) & (x.ios > self.config.entity_ios_thres) & (x.ios < 1 - self.config.entity_ios_thres)
            if 'mars_output' not in extra_output:
                extra_output['mars_output'] = mars_output = self.mars_loss.predict_mars_masks(extra_output['decoded_teacher'], self.collector['inpainting_mask_ori'], ios_thres=0.85)
            else:
                mars_output = extra_output['mars_output']
            hw_ori = mars_output.pred_masks_sigmoid.shape[2:]
            batch_size = mars_output.pred_masks_sigmoid.shape[0]
            def get_bboxes(mars_output):
                pos_indices = get_pos_indices(mars_output)
                pos_indices = torch.where(pos_indices)
                pred_masks = mars_output.pred_masks_sigmoid[pos_indices[0], pos_indices[1]] > 0.5
                if pred_masks.size(0) > 0:
                    bboxes = masks2bboxes(pred_masks)
                    bboxes = torch.cat([pos_indices[0][:, None].to(bboxes), bboxes], dim=1)
                else:
                    bboxes = torch.zeros((0, 5), dtype=torch.float, device='cuda')
                bbox_global = bboxes.new_tensor([0, 0, hw_ori[1] - 1, hw_ori[0] - 1])
                if self.config.max_entity_bboxes_num > 0:
                    if bboxes.size(0) > self.config.max_entity_bboxes_num:
                        indices = torch.randperm(bboxes.size(0))[:self.config.max_entity_bboxes_num]  # 随机打乱后取前 k 个
                        bboxes = bboxes[indices]
                bbox_global = torch.cat([torch.arange(batch_size, dtype=bbox_global.dtype, device=bbox_global.device)[:, None], bbox_global[None].repeat(batch_size, 1)], dim=1)
                bboxes = torch.cat([bboxes, bbox_global], dim=0)
                return bboxes
            boxes_teacher = get_bboxes(mars_output)
            mars_output = self.mars_loss.predict_mars_masks(extra_output['decoded_student'], self.collector['inpainting_mask_ori'], ios_thres=0.85)
            extra_output['mars_output_student'] = mars_output
            boxes_student = get_bboxes(mars_output)
            boxes_teacher[:, 0] += student_output.shape[0]
            entity_boxes = torch.cat([boxes_student, boxes_teacher], dim=0)
            discriminator_output = self.discriminator(
                noisy_sample.to(self.config.train_dtype),
                timestep,
                entity_boxes=entity_boxes,
                hw_ori=hw_ori,
                conditioning=conditioning,
            )
            extra_output["discriminator_output"] = discriminator_output
            extra_output["entity_boxes"] = entity_boxes
            extra_output["hw_ori"] = hw_ori
            predict_prob = discriminator_output.prob[..., 0].sigmoid()
            label = torch.cat([predict_prob.new_zeros([boxes_student.size(0)]), predict_prob.new_ones([boxes_teacher.size(0)])])
            bboxes_area_ratio = (entity_boxes[:, 3] - entity_boxes[:, 1]) * (entity_boxes[:, 4] - entity_boxes[:, 2]) / (hw_ori[0] * hw_ori[1])
            valid_G = (1 - label) * bboxes_area_ratio

            if self.gan_loss_type == "lsgan":
                if training_G:
                    loss_G = F.mse_loss(predict_prob, 1 - label, reduction='none') * valid_G
                    loss_G = loss_G.sum() / valid_G.sum().clamp(min=1e-4)
                    loss_D = 0
                else:
                    loss_D = 0.5 * F.mse_loss(predict_prob, label, reduction='none') * bboxes_area_ratio
                    loss_D = loss_D.sum() / bboxes_area_ratio.sum().clamp(min=1e-4)
                    loss_G = 0
            else:
                raise NotImplementedError
        if not self.config.train_D_on_non_pair_label and data_tag == "non_pair_label":
            loss_D = 0

        return [
            loss_G,
            loss_D,
        ]

    def _timestep_sampling(
        self, n_samples: int = 1, device="cpu", timestep_sampling="uniform"
    ) -> torch.Tensor:
        if timestep_sampling == "uniform":
            idx = self.prob.multinomial(n_samples, replacement=True).to(device)

            return torch.tensor(
                self.selected_timesteps, device=device, dtype=torch.long
            )[idx]

        elif timestep_sampling == "teacher":
            return torch.randint(
                0,
                self.teacher_noise_scheduler.config.num_train_timesteps,
                (n_samples,),
                device=device,
            )

    def _get_conditioning(
        self,
        batch: Dict[str, Any],
        ucg_keys: List[str] = None,
        set_ucg_rate_zero=False,
        *args,
        **kwargs,
    ):
        """
        Get the conditionings
        """
        if self.conditioner is not None:
            return self.conditioner(
                batch,
                ucg_keys=ucg_keys,
                set_ucg_rate_zero=set_ucg_rate_zero,
                vae=self.vae,
                *args,
                **kwargs,
            )
        else:
            return None

    def _scalings_for_boundary_conditions(self, timestep, sigma_data=0.5):
        """
        Compute the scalings for boundary conditions
        """
        c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
        c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
        return c_skip, c_out

    def _predicted_x_0(
        self,
        model_output,
        timesteps,
        sample,
        prediction_type,
        alphas,
        sigmas,
        input_sample,
    ):
        """
        Predict x_0 using the model output and the timesteps depending on the prediction type
        """
        if prediction_type == "epsilon":
            sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
            alphas = extract_into_tensor(alphas, timesteps, sample.shape)
            alpha_mask = alphas > 0
            alpha_mask = alpha_mask.reshape(-1)
            alpha_mask_0 = alphas == 0
            alpha_mask_0 = alpha_mask_0.reshape(-1)
            pred_x_0 = torch.zeros_like(sample)
            pred_x_0[alpha_mask] = (
                sample[alpha_mask] - sigmas[alpha_mask] * model_output[alpha_mask]
            ) / alphas[alpha_mask]
            pred_x_0[alpha_mask_0] = input_sample[alpha_mask_0]
        elif prediction_type == "v_prediction":
            sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
            alphas = extract_into_tensor(alphas, timesteps, sample.shape)
            pred_x_0 = alphas * sample - sigmas * model_output
        else:
            raise ValueError(
                f"Prediction type {prediction_type} currently not supported."
            )

        return pred_x_0

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_steps: int = 20,
        guidance_scale: float = 1.0,
        teacher_guidance_scale: float = 5.0,
        conditioner_inputs: Dict[str, Any] = None,
        uncond_conditioner_inputs: Dict[str, Any] = None,
        max_samples: int = None,
        verbose: bool = False,
        log_teacher_samples: bool = False,
        adapter_conditioning_scale: float = 1.0,
    ):
        """
        Sample from the model
        Args:
            z (torch.Tensor): Noisy latent vector
            num_steps: (int): Number of steps to sample
            guidance_scale (float): Guidance scale for classiffier-free guidance. If 1, no guidance. Default: 1.0
            conditioner_inputs (Dict[str, Any]): inputs to the conditioners
            uncond_conditioner_inputs (Dict[str, Any]): inputs to the conditioner for CFG e.g. negative prompts.
            max_samples (Optional[int]): Maximum number of samples to generate. Default: None, all samples are generated
            verbose (bool): Whether to print progress bar. Default: True
            adapter_conditioning_scale (float): Adapter conditioning scale. Default: 1.0
        """

        self.teacher_noise_scheduler.set_timesteps(num_steps)

        # Set the sampling noise scheduler to the right number of timesteps for inference
        try:
            self.sampling_noise_scheduler.set_timesteps(
                timesteps=self.teacher_noise_scheduler.timesteps
            )
        except:
            self.sampling_noise_scheduler.set_timesteps(num_steps)

        sample = z

        # Get conditioning
        conditioning = self._get_conditioning(
            conditioner_inputs, set_ucg_rate_zero=True, device=z.device
        )

        # Get unconditional conditioning
        if uncond_conditioner_inputs is not None:
            unconditional_conditioning = self._get_conditioning(
                uncond_conditioner_inputs, set_ucg_rate_zero=True, device=z.device
            )
        else:
            unconditional_conditioning = self._get_conditioning(
                conditioner_inputs, ucg_keys=self.ucg_keys, device=z.device
            )

        if max_samples is not None:
            sample = sample[:max_samples]

            if conditioning:
                conditioning["cond"] = {
                    k: v[:max_samples] for k, v in conditioning["cond"].items()
                }
                unconditional_conditioning["cond"] = {
                    k: v[:max_samples]
                    for k, v in unconditional_conditioning["cond"].items()
                }

        # Compute the T2I adapter features
        if self.adapter:
            down_intrablock_additional_residuals = self.adapter(
                conditioner_inputs[self.adapter_input_key]
            )
            for k, v in enumerate(down_intrablock_additional_residuals):
                down_intrablock_additional_residuals[k] = v * adapter_conditioning_scale

        else:
            down_intrablock_additional_residuals = None

        sample_init = sample
        sample = sample * self.sampling_noise_scheduler.init_noise_sigma
        for i, t in tqdm(
            enumerate(self.sampling_noise_scheduler.timesteps), disable=not verbose
        ):
            denoiser_input = self.sampling_noise_scheduler.scale_model_input(sample, t)

            # Predict noise level using denoiser using conditionings
            cond_noise_pred = self.student_denoiser(
                sample=denoiser_input,
                timestep=t.to(z.device).repeat(denoiser_input.shape[0]),
                **conditioning,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            )

            # Predict noise level using denoiser using unconditional conditionings
            uncond_noise_pred = self.student_denoiser(
                sample=denoiser_input,
                timestep=t.to(z.device).repeat(denoiser_input.shape[0]),
                **unconditional_conditioning,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            )

            # Make CFG
            noise_pred = (
                guidance_scale * cond_noise_pred
                + (1 - guidance_scale) * uncond_noise_pred
            )

            # Make one step on the reverse diffusion process
            sample = self.sampling_noise_scheduler.step(
                noise_pred, t, sample, return_dict=False
            )[0]

        if self.vae is not None:
            decoded_sample = self.vae.decode(sample)
        else:
            decoded_sample = sample

        decoded_sample_ref = None

        if log_teacher_samples:
            self.teacher_sampling_noise_scheduler.set_timesteps(num_steps)

            sample_ref = (
                sample_init * self.teacher_sampling_noise_scheduler.init_noise_sigma
            )

            for i, t in tqdm(
                enumerate(self.teacher_sampling_noise_scheduler.timesteps),
                disable=not verbose,
            ):
                denoiser_input_ref = (
                    self.teacher_sampling_noise_scheduler.scale_model_input(
                        sample_ref, t
                    )
                )

                cond_noise_pred_ref = self.teacher_denoiser(
                    sample=denoiser_input_ref,
                    timestep=t.to(z.device).repeat(denoiser_input_ref.shape[0]),
                    **conditioning,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                )
                uncond_noise_pred_ref = self.teacher_denoiser(
                    sample=denoiser_input_ref,
                    timestep=t.to(z.device).repeat(denoiser_input_ref.shape[0]),
                    **unconditional_conditioning,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                )

                noise_pred_ref = (
                    teacher_guidance_scale * cond_noise_pred_ref
                    + (1 - teacher_guidance_scale) * uncond_noise_pred_ref
                )
                sample_ref = self.teacher_sampling_noise_scheduler.step(
                    noise_pred_ref, t, sample_ref, return_dict=False
                )[0]

                if self.vae is not None:
                    decoded_sample_ref = self.vae.decode(sample_ref)
                else:
                    decoded_sample_ref = sample_ref

        return decoded_sample, decoded_sample_ref

    def log_samples(
        self,
        batch: Dict[str, Any],
        input_shape: Tuple[int, int, int] = None,
        guidance_scale: float = 1.0,
        teacher_guidance_scale: float = 5.0,
        max_samples: int = 8,
        num_steps: Union[int, List[int]] = 20,
        device="cpu",
        log_teacher_samples=False,
        conditioner_inputs: Dict = None,
        conditioner_uncond_inputs: Dict = None,
        adapter_conditioning_scale: float = 1.0,
    ):

        if isinstance(num_steps, int):
            num_steps = [num_steps]

        logs = {}

        N = max_samples

        if batch is not None:
            max_conditioning_samples = min([len(batch[key]) for key in batch])
            N = min(N, max_conditioning_samples)

        if conditioner_inputs is not None:
            max_conditioning_samples = min(
                [len(conditioner_inputs[key]) for key in conditioner_inputs]
            )
            conditioner_inputs_ = {
                k: v.to(device)
                for k, v in conditioner_inputs.items()
                if isinstance(v, torch.Tensor)
            }
            conditioner_inputs.update(conditioner_inputs_)
            batch.update(conditioner_inputs)
            N = min(N, max_conditioning_samples)

        if conditioner_uncond_inputs is not None:
            max_conditioning_samples = min(
                [
                    len(conditioner_uncond_inputs[key])
                    for key in conditioner_uncond_inputs
                ]
            )
            conditioner_uncond_inputs_ = {
                k: v.to(device)
                for k, v in conditioner_uncond_inputs.items()
                if isinstance(v, torch.Tensor)
            }
            conditioner_uncond_inputs.update(conditioner_uncond_inputs_)
            batch_uncond = deepcopy(batch)
            batch_uncond.update(conditioner_uncond_inputs)
            N = min(N, max_conditioning_samples)
        else:
            batch_uncond = None

        # infer input shape based on VAE configuration if not passed
        if input_shape is None:
            if self.vae is not None:
                # get input pixel size of the vae
                input_shape = batch[self.vae.config.input_key].shape[2:]
                # rescale to latent size
                input_shape = (
                    self.vae.latent_channels,
                    input_shape[0] // self.vae.downsampling_factor,
                    input_shape[1] // self.vae.downsampling_factor,
                )
            else:
                raise ValueError(
                    "input_shape must be passed when no VAE is used in the model"
                )

        for num_step in num_steps:
            # Log samples
            z = torch.randn(N, *input_shape).to(device)

            logging.debug(
                f"Sampling {N} samples: steps={num_step}, guidance_scale={guidance_scale}"
            )
            samples, samples_ref = self.sample(
                z,
                num_steps=num_step,
                conditioner_inputs=batch,
                uncond_conditioner_inputs=batch_uncond,
                guidance_scale=guidance_scale,
                teacher_guidance_scale=teacher_guidance_scale,
                max_samples=N,
                log_teacher_samples=log_teacher_samples,
                adapter_conditioning_scale=adapter_conditioning_scale,
            )

            logs[
                f"samples_{num_step}_steps/{self.sampling_noise_scheduler.__class__.__name__}_{guidance_scale}_cfg/student"
            ] = samples

            if samples_ref is not None:
                logs[
                    f"samples_{num_step}_steps/{self.teacher_sampling_noise_scheduler.__class__.__name__}_{teacher_guidance_scale}_cfg/teacher"
                ] = samples_ref

        return logs

    def feed_models(self, models):
        self.student_denoiser, self.discriminator = models

class Discriminator(nn.Module):
    def __init__(self, discriminator_feature_dim=64, color_dim=1280):
        super().__init__()
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Conv2d(color_dim, discriminator_feature_dim, 3, 1, 1),
            nn.SiLU(True),
            nn.Conv2d(
                discriminator_feature_dim,
                discriminator_feature_dim * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.SiLU(True),
            nn.GroupNorm(4, discriminator_feature_dim * 2),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(discriminator_feature_dim * 2, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        prob = self.discriminator(x)
        return prob


if __name__ == "__main__":
    from unhcv.common.utils import find_path
    import os
    sd_path = find_path("model/stable-diffusion-v1-5-inpainting")

    class ErasureDenoiser:

        @staticmethod
        def forward(self, sample, *args, **kwargs):
            masked_latent = self.collector["masked_latent"]
            inpainting_mask = self.collector["inpainting_mask"]
            sample = torch.cat((sample, inpainting_mask, masked_latent), dim=1)
            return self._forward(sample, *args, **kwargs).sample

    collector = {}
    vae = VAE(os.path.join(sd_path, "vae"), return_type='mean').cuda()
    student_unet = UNet2DConditionModel.from_pretrained(os.path.join(sd_path, "unet"))
    teacher_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        sd_path,
        subfolder="scheduler",
        timestep_spacing="trailing",
    )
    sampling_scheduler = LCMScheduler.from_pretrained(
        sd_path,
        subfolder="scheduler",
        timestep_spacing="trailing",
    )
    teacher_sampling_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        sd_path,
        subfolder="scheduler",
    )

    model = FlashDistill(
        dict(num_iterations_per_K=(3, 3, 3, 1000)),
        student_denoiser=student_unet,
        teacher_denoiser=None,
        teacher_noise_scheduler=teacher_scheduler,
        sampling_noise_scheduler=sampling_scheduler,
        teacher_sampling_noise_scheduler=teacher_sampling_scheduler,
        discriminator=Discriminator(),
        collector=collector,
        vae=vae
    )

    model: FlashDistill = model.cuda()
    image_size = (64, 32)
    z = torch.randn(1, 4, *image_size).cuda()
    condition = torch.randn(1, 5, *image_size).cuda()

    conditioning = dict(encoder_hidden_states=torch.randn(1, 77, 768).cuda())
    with torch.no_grad():
        for _ in range(100):
            collector["masked_latent"] = torch.randn(1, 4, *image_size).cuda()
            collector["inpainting_mask"] = torch.randn(1, 1, *image_size).cuda()
            output = model.forward(z, conditioning=conditioning, unconditional_conditioning=conditioning, step=0)
