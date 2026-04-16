import os
import torch

from diffusers.models.attention_processor import AttnProcessor
from safetensors import safe_open
from torch import nn
from transformers import CLIPVisionModelWithProjection

from ip_adapter.attention_processor import IPAttnProcessor2_0, AttnProcessor2_0
from ip_adapter.ip_adapter import ImageProjModel
from unhcv.common.utils import find_path


class IPAdapter(nn.Module):
    def __init__(self, denoiser, num_tokens=4, clip_dim=1024, image_encoder_path=None):
        super().__init__()
        self.denoiser = denoiser
        self.num_tokens = num_tokens
        self.set_ip_adapter(denoiser)

        # load image encoder
        if image_encoder_path is not None:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        # self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj(denoiser, clip_dim=clip_dim)

    def init_proj(self, denoiser, clip_dim):
        image_proj_model = ImageProjModel(cross_attention_dim=denoiser.config.cross_attention_dim,
                                          clip_embeddings_dim=clip_dim, clip_extra_context_tokens=self.num_tokens, )
        return image_proj_model

    def set_ip_adapter(self, denoiser):
        attn_procs = {}
        for name in denoiser.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else denoiser.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = denoiser.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(denoiser.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = denoiser.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                )
        denoiser.set_attn_processor(attn_procs)

    def load_ip_adapter(self, ip_ckpt):
        if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.denoiser.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    def set_scale(self, scale):
        for attn_processor in self.denoiser.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor2_0):
                attn_processor.scale = scale

    def get_image_embeds(self, clip_image):
        clip_image_embeds = self.image_encoder(clip_image).image_embeds
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds


if __name__ == "__main__":
    pass