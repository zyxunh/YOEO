import torch.nn as nn
import torch
from diffusers import AutoencoderKL


__all__ = ['VAE']


class VAE(nn.Module):
    def __init__(self, pretrained_model_name_or_path=None, config=None, return_type="sample", torch_dtype=torch.float32):
        super().__init__()
        self.model: AutoencoderKL = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)
        self.return_type = return_type

    def encode(self, x):
        latent_dist = self.model.encode(x).latent_dist
        if self.return_type == "sample":
            latents = latent_dist.sample()
        elif self.return_type == "mean":
            latents = latent_dist.mean
        else:
            raise NotImplementedError(f"Unsupported return type: {self.return_type}")
        latents = latents * self.model.config.scaling_factor
        return latents

    def decode(self, x, clip=True):
        image = self.model.decode(x / self.model.config.scaling_factor).sample
        if clip:
            image = image.clip(-1, 1)
        return image

if __name__ == '__main__':
    vae = VAE(pretrained_model_name_or_path="/home/yixing/model/stable-diffusion-v1-5-inpainting/vae")
    print(vae)
    print(vae.model)
    input = torch.randn(1, 3, 64, 64).clamp(min=-1, max=1)
    latents = vae.encode(input)
    print(latents.shape)
    image = vae.decode(latents)
    print(image.shape)
    breakpoint()