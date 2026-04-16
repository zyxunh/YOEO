import os

from unhcv.common import write_im
from unhcv.common.image import visual_tensor
from unhcv.common.utils import find_path, attach_home_root

from diffusers import UNet2DConditionModel, DDIMScheduler
import torch

from unhcv.nn.models.diffusion import VAE, TextEncoder, build_scheduler, ode, ODE

root = find_path("model/stable-diffusion-v1-5")
scheduler = build_scheduler(os.path.join(root, "scheduler", "scheduler_config.json"))
scheduler = DDIMScheduler.from_pretrained(os.path.join(root, "scheduler", "scheduler_config.json"))
vae = VAE(pretrained_model_name_or_path=os.path.join(root, "vae"), return_type="mean", torch_dtype=torch.float16).cuda()
text_encoder = TextEncoder(pretrained_model_name_or_path=os.path.join(root, "text_encoder"),
                           tokenizer_name_or_path=os.path.join(root, "tokenizer"), torch_dtype=torch.float16).cuda()
denoiser: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(os.path.join(root, "unet")).to(device="cuda", dtype=torch.float16)

prompt = "A dog"
neg_prompt = ""
encoder_hidden_states = text_encoder(prompt)
neg_encoder_hidden_states = text_encoder(neg_prompt)
generator = torch.Generator(device="cuda").manual_seed(1234)
noise = torch.randn((1, 4, 64, 64), generator=generator, device="cuda", dtype=torch.float16)

output_latent = ODE.ode(scheduler, denoiser, noise, condition=dict(encoder_hidden_states=encoder_hidden_states), un_condition=dict(encoder_hidden_states=neg_encoder_hidden_states), steps=20, generator=generator, guidance_scale=7.5)
output_image = vae.decode(output_latent)
output_image = visual_tensor(output_image, max_value=1, min_value=-1, reverse=True)
write_im(attach_home_root("train_outputs/test.jpg"), output_image)
breakpoint()