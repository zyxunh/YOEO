import dataclasses
import os
from functools import partial
from typing import Union

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from torch import dtype
from transformers import CLIPImageProcessor, CLIPTokenizer

from unhcv.common import CfgNode
from unhcv.common.utils import find_path
from unhcv.nn.models.diffusion import DiffusionTools, VAE, ODE, TextEncoder
from unhcv.nn.utils import freeze_model, wrap_no_grad, load_checkpoint

__all__ = ["ErasureModel", "ErasureModelConfig"]


@dataclasses.dataclass
class ErasureModelConfig(CfgNode):
    denoise_model_name = "default"
    denoise_pretrained_path: str = None
    denoise_pretrained_root: str = "model/stable-diffusion-v1-5-inpainting"
    denoise_pretrained_model: str = None
    train_dtype: dtype = torch.float
    ode_steps: int = 10
    null_condition: bool = True
    inpaint_pad_value: float = 0

    def __post_init__(self):
        self.denoise_pretrained_root = find_path(self.denoise_pretrained_root)
        if self.denoise_pretrained_path is None:
            self.denoise_pretrained_path = os.path.join(self.denoise_pretrained_root, 'unet')
        self.denoise_pretrained_path = find_path(self.denoise_pretrained_path)
        self.denoise_pretrained_model = find_path(self.denoise_pretrained_model)


class ErasureModel(nn.Module):
    def __init__(self, config):
        super(ErasureModel, self).__init__()
        self.config = config = ErasureModelConfig.from_other(config)
        self.ddpm_scheduler = DiffusionTools.build_scheduler(config.denoise_pretrained_root)
        self.ddim_scheduler = DiffusionTools.build_scheduler(config.denoise_pretrained_root, name="DDIM")
        self.vae: VAE = VAE(os.path.join(config.denoise_pretrained_root, "vae")).to(device="cuda", dtype=self.config.train_dtype)

        if self.config.denoise_model_name == "default":
            text_encoder = TextEncoder(
                pretrained_model_name_or_path=os.path.join(config.denoise_pretrained_root, "text_encoder"),
                tokenizer_name_or_path=os.path.join(config.denoise_pretrained_root, "tokenizer"),
                torch_dtype=config.train_dtype).cuda()
            if self.config.null_condition:
                with torch.no_grad():
                    text_feature = text_encoder([""])
                del text_encoder
                self.text_feature = text_feature

            denoiser: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(self.config.denoise_pretrained_path)
            if config.denoise_pretrained_model is not None:
                load_checkpoint(denoiser, config.denoise_pretrained_model, log_missing_keys=True)
        elif self.config.denoise_model_name == "smart_eraser":
            raise NotImplementedError

        self.denoiser = denoiser.cuda()
        self.ode_steps = config.ode_steps

        freeze_model(self.vae)

    @wrap_no_grad
    def image_to_latent(self, image):
        return self.vae.encode(image)

    @wrap_no_grad
    def latent_to_image(self, latent):
        return self.vae.decode(latent)

    def denoise(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor, inpainting_mask, masked_image_latent):
        """
        单步
        Args:
            sample:
            timestep:
            encoder_hidden_states:
        """
        sample = torch.cat([sample, inpainting_mask.to(sample), masked_image_latent.to(sample)], dim=1)
        return self.denoiser(sample, timestep, encoder_hidden_states.to(sample))

    def ode(self, sample, encoder_hidden_states, inpainting_mask, masked_image_latent, requires_grad=True):
        denoise = partial(self.denoise, inpainting_mask=inpainting_mask, masked_image_latent=masked_image_latent)
        output = ODE.ode(self.ddim_scheduler, denoise, sample, condition=dict(encoder_hidden_states=encoder_hidden_states),
                steps=self.ode_steps, requires_grad=requires_grad)
        return output

    def feed_models(self, models):
        self.denoiser = models


if __name__ == "__main__":
    erasure_model = ErasureModel(None)
    sample = torch.randn(2, 4, 64, 64).cuda()
    encoder_hidden_states = torch.randn(2, 77, 768).cuda()
    inpainting_mask = torch.randn(2, 1, 64, 64).cuda()
    masked_image_latent = torch.randn(2, 4, 64, 64).cuda()
    erasure_model.ode(sample, encoder_hidden_states, inpainting_mask, masked_image_latent)