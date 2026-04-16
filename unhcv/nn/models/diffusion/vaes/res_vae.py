from functools import partial

import torch
from torch import nn
from unhcv.nn.models.diffusion import VAE
import os
import requests
from tqdm import tqdm
from diffusers import DDPMScheduler, AutoencoderKL

__all__ = ["ResVAE", "ResVAEDiffusion"]

from unhcv.nn.utils import clone_conv


class ResVAE(VAE):

    @staticmethod
    def make_1step_sched(device):
        noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
        noise_scheduler_1step.set_timesteps(1, device=device)
        noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.to(device)
        return noise_scheduler_1step

    def custom_init(self, in_channels: int=None):
        vae = self.model
        collector = {}
        # add the skip connection convs
        vae.decoder.collector = vae.encoder.collector = collector
        vae.encoder.forward = self.my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = self.my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
        torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
        vae.decoder.ignore_skip = False
        vae.decoder.gamma = 1

        if in_channels is not None:
            vae.encoder.conv_in = clone_conv(vae.encoder.conv_in, resume_weight=True, in_channels=in_channels)

    @staticmethod
    def my_vae_encoder_fwd(self: AutoencoderKL, sample):
        sample = self.conv_in(sample)
        l_blocks = []
        # down
        for down_block in self.down_blocks:
            l_blocks.append(sample)
            sample = down_block(sample)
        # middle
        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        self.collector["current_down_blocks"] = l_blocks
        return sample

    @staticmethod
    def my_vae_decoder_fwd(self: AutoencoderKL, sample, latent_embeds=None):
        sample = self.conv_in(sample)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)
        if not self.ignore_skip:
            incoming_skip_acts = self.collector.pop("current_down_blocks")  # self.current_down_blocks
            skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
            # up
            for idx, up_block in enumerate(self.up_blocks):
                skip_in = skip_convs[idx](incoming_skip_acts[::-1][idx] * self.gamma)
                # add skip
                sample = sample + skip_in
                sample = up_block(sample, latent_embeds)
        else:
            for idx, up_block in enumerate(self.up_blocks):
                sample = up_block(sample, latent_embeds)
        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class ResVAEDiffusion(nn.Module):
    def __init__(self, unet, vae):
        super(ResVAEDiffusion, self).__init__()
        self.unet = unet
        self.vae:ResVAE = vae

    def forward(self, x, timestep, encoder_hidden_states):
        latent = self.vae.encode(x)
        latent = self.unet(latent, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        output = self.vae.decode(latent, clip=False)
        return output
