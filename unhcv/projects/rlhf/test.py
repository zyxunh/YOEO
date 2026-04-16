from functools import partial

import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from torchvision.transforms.functional import pil_to_tensor

from unhcv.common import write_im
from unhcv.common.image import visual_tensor
from unhcv.common.utils import attach_home_root
from unhcv.datasets.common_datasets import SizeBucket
from unhcv.datasets.transforms.torchvision_transforms import RandomResizedCrop
from unhcv.nn.models.diffusion import DiffusionTools, ODE
from unhcv.projects.rlhf import ErasureModelConfig, ErasureModel


def main():
    denoise_pretrained_path = "model path"
    config = ErasureModelConfig(denoise_pretrained_model=denoise_pretrained_path)
    evaluation_model = ErasureModel(config)
    lcm_scheduler = DiffusionTools.build_scheduler(config.denoise_pretrained_root,
                                                                 name="CustomLCM", timestep_spacing="trailing")
    size_bucket = SizeBucket()

    mask = PIL.Image.open("00001_mask.png")
    image = PIL.Image.open("00001.jpg")
    hw = size_bucket.match((image.height, image.width))
    ratio = image.width / image.height
    random_resized_crop = RandomResizedCrop(size=hw, ratio=(ratio, ratio), scale=(1, 1))
    image, mask = random_resized_crop([image, mask], interpolations=("bicubic", "nearest-exact"))
    image = pil_to_tensor(image)[None]
    mask = pil_to_tensor(mask)[None]

    image = image.float().cuda() / 127.5 - 1
    mask = mask.float().cuda() / 255
    image_masked = image * (1 - mask)
    masked_image_latent = evaluation_model.image_to_latent(image_masked)
    mask = F.interpolate(mask, size=masked_image_latent.shape[-2:], mode="nearest-exact")
    noise = torch.randn_like(masked_image_latent)
    encoder_hidden_states = evaluation_model.text_feature
    denoiser = partial(evaluation_model.denoise, inpainting_mask=mask, masked_image_latent=masked_image_latent)
    output_latent_ode = ODE.ode(lcm_scheduler, denoiser, noise,
                                condition=dict(encoder_hidden_states=encoder_hidden_states),
                                generator=torch.Generator(device="cuda").manual_seed(1234),
                                guidance_scale=1,
                                steps=2)
    result_image = evaluation_model.latent_to_image(output_latent_ode)
    result_image = visual_tensor(result_image, max_value=1, min_value=-1, reverse=False)
    write_im(attach_home_root("show/result.jpg"), result_image)
    return result_image


if __name__ == "__main__":
    main()
