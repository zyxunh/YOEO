import torch
from diffusers import UNet2DConditionModel
from diffusers.models.upsampling import Upsample2D

from unhcv.common.utils import find_path
from unhcv.nn.models.diffusion.controlnet import UnetSiamese
from unhcv.nn.models.diffusion.controlnet.utils import BrushNet

model = UNet2DConditionModel.from_pretrained(find_path("model/stable-diffusion-v1-5/unet"))
model = BrushNet.init(model)

model = UnetSiamese.init(model)
model_siamese = UnetSiamese.init_siamese(model)
noise = torch.randn(1, 4, 64, 64)
timesteps = 5
encoder_hidden_states = torch.randn(1, 77, 768)
output = model.forward(noise, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=True)
k = list(output.up_block_res_samples)
output1 = model_siamese.forward(timestep=timesteps, encoder_hidden_states=encoder_hidden_states, down_block_res_samples=list(output.down_block_res_samples), mid_block_add_sample=output.mid_block_res_sample, up_block_add_samples=k)
upsample = Upsample2D(channels=320, use_conv_transpose=True, out_channels=320 // 4, norm_type="ln_norm")
k = upsample(output1.last_feature)
breakpoint()
pass
