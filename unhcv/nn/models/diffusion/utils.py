import diffusers
import importlib
from einops import repeat

from typing import Union, Dict, Type, Tuple

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel, PNDMScheduler, DDPMScheduler, LCMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from torch import nn

from unhcv.common.utils import obj_load


__all__ = ["add_cond_wrapper", "ode", "build_scheduler", "ConditioningWrappedUnet", "ODE", "MergeWeight",
           "DiffusionTools"]

from unhcv.nn.utils import wrap_no_grad


class CondWrapper:

    def forward(
        self,
        sample: torch.Tensor,
        *args,
        conditioning: Dict[str, torch.Tensor] = None,
        **kwargs
    ):
        class_labels = conditioning.get("vector", None)
        crossattn = conditioning.get("crossattn", None)
        concat = conditioning.get("concat", None)

        # concat conditioning
        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        return self.model(sample, *args, encoder_hidden_states=crossattn, class_labels=class_labels, **kwargs).sample


def add_cond_wrapper(cls) -> Type[UNet2DConditionModel]:
    class WrappedClass(cls):
        def forward(
                self,
                sample: torch.Tensor,
                *args,
                conditioning: Dict[str, torch.Tensor] = None,
                **kwargs):
            """

            Args:
                sample:
                *args:
                conditioning: dict(encoder_hidden_states, concat, class_labels, conv_in_residual)
                **kwargs:

            Returns:

            """
            conditioning = conditioning if conditioning is not None else {}
            conditioning = conditioning.copy()
            concat = conditioning.pop("concat", None)

            # concat conditioning
            if concat is not None:
                if sample is None:
                    sample = concat
                else:
                    sample = torch.cat([sample, concat], dim=1)

            return super().forward(sample, *args, **conditioning, **kwargs)

    return WrappedClass

class ConditioningWrappedUnet(UNet2DConditionModel):
    def forward(self, sample: torch.Tensor, *args, conditioning: Dict[str, torch.Tensor] = None, **kwargs) -> Union[UNet2DConditionOutput, Tuple]:
        pass

class ODE:

    @classmethod
    @wrap_no_grad
    def ode(cls, scheduler: DDIMScheduler, denoiser, noisy_sample, condition, un_condition=None,
        guidance_scale=1, start_idx=0, end_idx=None, steps=None, generator=None, un_condition_parallel=False, denoise_pre_func=None,
        denoise_after_func= lambda x: x if isinstance(x, torch.Tensor) else x.sample, return_original_sample=False) -> torch.Tensor:
        """

        Args:
            scheduler:
            denoiser:
            noisy_sample:
            condition: dict(encoder_hidden_states, concat)
            un_condition:
            guidance_scale:
            start_idx:

        Returns:

        """
        # if steps is not None and scheduler.num_inference_steps != steps:
        scheduler.set_timesteps(steps)
        noisy_sample = noisy_sample * scheduler.init_noise_sigma
        timesteps = scheduler.timesteps
        if end_idx is None:
            timesteps = timesteps[start_idx:]
        else:
            timesteps = timesteps[start_idx:end_idx]

        for t in timesteps:
            if un_condition_parallel and guidance_scale > 1:
                noisy_sample_origin = noisy_sample
                noisy_sample = repeat(noisy_sample, 'b c h w -> (n b) c h w', n=2)

            if denoise_pre_func is not None:
                noisy_sample = noisy_sample_ = denoise_pre_func(noisy_sample)
                if not isinstance(noisy_sample, torch.Tensor):
                    noisy_sample = noisy_sample_.pop("sample")
                    condition.update(noisy_sample_)

            timestep = torch.tensor([t], device=noisy_sample.device).repeat(noisy_sample.shape[0])

            noisy_sample_ = scheduler.scale_model_input(noisy_sample, t)

            # Denoise sample
            noise_pred = cond_noise_pred = denoise_after_func(denoiser(
                sample=noisy_sample_,
                timestep=timestep,
                **condition
            ))

            if guidance_scale > 1:
                if un_condition_parallel:
                    uncond_noise_pred, cond_noise_pred = torch.chunk(noise_pred, 2, dim=0)
                    noisy_sample = noisy_sample_origin
                else:
                    uncond_noise_pred = denoise_after_func(denoiser(
                        sample=noisy_sample_,
                        timestep=timestep,
                        **un_condition,
                    ))

                # Make CFG
                noise_pred = (
                        guidance_scale * cond_noise_pred
                        + (1 - guidance_scale) * uncond_noise_pred
                )

            # Make one step on the reverse diffusion process
            if return_original_sample:
                noisy_sample, original_sample = scheduler.step(noise_pred, t, noisy_sample, return_dict=False)[:2]
            else:
                noisy_sample = scheduler.step(noise_pred, t, noisy_sample, return_dict=False)[0]

            noisy_sample = noisy_sample.to(noise_pred)
        if return_original_sample:
            return noisy_sample, original_sample

        return noisy_sample

def ode(scheduler: DDIMScheduler, denoiser, noisy_sample, conditioning, unconditional_conditioning=None,
        guidance_scale=1, start_idx=0, steps=None, generator=None) -> torch.Tensor:
    """

    Args:
        scheduler:
        denoiser:
        noisy_sample:
        conditioning: dict(encoder_hidden_states, concat)
        unconditional_conditioning:
        guidance_scale:
        start_idx:

    Returns:

    """

    if steps is not None and scheduler.num_inference_steps != steps:
        scheduler.set_timesteps(steps)
    noisy_sample = noisy_sample * scheduler.init_noise_sigma
    for t in scheduler.timesteps[start_idx:]:
        timestep = torch.tensor([t], device=noisy_sample.device).repeat(noisy_sample.shape[0])
    
        noisy_sample_ = scheduler.scale_model_input(noisy_sample, t)
    
        # Denoise sample
        noise_pred = cond_noise_pred = denoiser(
            sample=noisy_sample_,
            timestep=timestep,
            conditioning=conditioning,
        ).sample

        if guidance_scale > 1:
            uncond_noise_pred = denoiser(
                sample=noisy_sample_,
                timestep=timestep,
                conditioning=unconditional_conditioning,
            ).sample

            # Make CFG
            noise_pred = (
                    guidance_scale * cond_noise_pred
                    + (1 - guidance_scale) * uncond_noise_pred
            )
    
        # Make one step on the reverse diffusion process
        noisy_sample = scheduler.step(noise_pred, t, noisy_sample, return_dict=False)[0]

        noisy_sample = noisy_sample.to(cond_noise_pred)

    return noisy_sample


def extract_into_tensor(a, t, x_dim):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (x_dim - 1)))


def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    sample_dim = sample.dim()
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample_dim)
        alphas = extract_into_tensor(alphas, timesteps, sample_dim)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample_dim)
        alphas = extract_into_tensor(alphas, timesteps, sample_dim)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def build_scheduler(scheduler_name_or_path) -> DDIMScheduler:
    config = obj_load(scheduler_name_or_path)
    class Scheduler(getattr(diffusers, "{}".format(config["_class_name"]))):
        pass
    # return Scheduler.from_pretrained(scheduler_name_or_path)
    return getattr(diffusers, "{}".format(config["_class_name"])).from_pretrained(scheduler_name_or_path)


class MergeWeight:

    @classmethod
    def merge_one_weight(cls, weight_0, weight_1, scale):
        if weight_0.shape != weight_1.shape:
            weight_1 = weight_1 * scale
            weight_0_new = weight_0.clone()
            weight_0_new[:, :min(weight_1.size(1), weight_0.size(1))] = (
                    weight_0_new[:, :min(weight_1.size(1), weight_0.size(1))] + weight_1)
        else:
            weight_0_new = weight_0 + weight_1 * scale
        return weight_0_new

    @classmethod
    def merge(cls, weight_dict_0, weight_dict_1, weight_dict_1_base=None, scale=0.5, scale_diff=1):
        weight_dict_0_update = {}
        if weight_dict_1_base is not None:
            weight_dict_1_diff = {}
            weight_dict_1_new = {}
            for key in weight_dict_1.keys():
                if key in weight_dict_1_base:
                    weight_dict_1_diff[key] = weight_dict_1[key] - weight_dict_1_base[key]
                else:
                    weight_dict_1_new[key] = weight_dict_1[key]

            for key in weight_dict_1_diff.keys():
                if key in weight_dict_0:
                    weight_dict_0_update[key] = cls.merge_one_weight(weight_dict_0[key], weight_dict_1_diff[key], scale_diff)
                else:
                    weight_dict_0_update[key] = weight_dict_1[key]
        else:
            weight_dict_1_new = weight_dict_1

        for key in weight_dict_1_new.keys():
            if key in weight_dict_0:
                assert weight_dict_1_base is None
                weight_dict_0_update[key] = weight_dict_0[key] * (1 - scale) + weight_dict_1_new[key] * scale
            else:
                weight_dict_0_update[key] = weight_dict_1_new[key]

        for key in weight_dict_0.keys():
            if key not in weight_dict_0_update:
                weight_dict_0_update[key] = weight_dict_0[key]

        return weight_dict_0_update


class DiffusionTools:

    @staticmethod
    def generate_noise(tensor, shape=None):
        return torch.randn_like(tensor)

    @staticmethod
    def generate_fixed_noise(tensor, shape=None, seed=1234, dtype=None, device='cuda'):
        if shape is None:
            shape = tensor.shape
            dtype = tensor.dtype
            device = tensor.device
        return torch.randn(shape, dtype=dtype, device=device, generator=torch.Generator(device="cuda").manual_seed(seed))

    @staticmethod
    def add_noise(ddpm_noise_scheduler, latents):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, ddpm_noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = ddpm_noise_scheduler.add_noise(latents, noise, timesteps)

        if ddpm_noise_scheduler.config.prediction_type == "epsilon":
            unet_target = noise
        elif ddpm_noise_scheduler.config.prediction_type == "v_prediction":
            unet_target = ddpm_noise_scheduler.get_velocity(latents, noise, timesteps)
        elif ddpm_noise_scheduler.config.prediction_type == "sample":
            unet_target = latents
        else:
            raise ValueError(f"Unknown prediction type {ddpm_noise_scheduler.config.prediction_type}")
        return noisy_latents, unet_target, timesteps

    @classmethod
    def build_scheduler(cls, path, name="DDPM", scheduler=None, **kwargs) -> DDPMScheduler:
        from unhcv.third_party.distill.utils import CustomLCMScheduler
        scheduler_name_dict = dict(DDPM=DDPMScheduler, DDIM=DDIMScheduler, LCM=LCMScheduler, CustomLCM=CustomLCMScheduler)
        if scheduler is None:
            scheduler = scheduler_name_dict[name]
        noise_scheduler = scheduler.from_pretrained(path, subfolder="scheduler", **kwargs)
        return noise_scheduler

    @staticmethod
    def delete_noise(scheduler: DDPMScheduler, noisy_latents: torch.Tensor, noise_pred: torch.Tensor, timestep: torch.IntTensor):
        # 2. compute alphas, betas
        if not isinstance(timestep, torch.Tensor):
            timestep = noisy_latents.new_tensor([timestep], dtype=torch.long)
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_t = alpha_prod_t[:, None, None, None]
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (noisy_latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            pred_epsilon = noise_pred
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = noise_pred
            pred_epsilon = (noisy_latents - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * noise_pred
            pred_epsilon = (alpha_prod_t**0.5) * noise_pred + (beta_prod_t**0.5) * noisy_latents
        else:
            raise ValueError(
                f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        return pred_original_sample


if __name__ == "__main__":
    scheduler = build_scheduler("/home/yixing/model/stable-diffusion-v1-5-inpainting/scheduler/scheduler_config.json")
    # scheduler = Scheduler("/home/yixing/model/stable-diffusion-v1-5-inpainting/scheduler/scheduler_config.json")
    breakpoint()
    pass