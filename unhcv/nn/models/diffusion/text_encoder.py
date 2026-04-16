import os

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from unhcv.common.utils import obj_load


__all__ = ["TextEncoder"]


class TextEncoder(torch.nn.Module):

    def __init__(self, pretrained_model_name_or_path, tokenizer_name_or_path, padding="max_length", torch_dtype=torch.float32):
        super().__init__()
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)
        tokenizer_config = obj_load(os.path.join(tokenizer_name_or_path, "tokenizer_config.json"))
        self.tokenizer: CLIPTokenizer = eval(tokenizer_config['tokenizer_class']).from_pretrained(tokenizer_name_or_path)
        self.padding = padding

    def tokenize(self, text):
        text_inputs = self.tokenizer(
                text,
                padding=self.padding,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        return text_inputs

    def encode(self, text, clip_skip=None, text_input_ids=None):
        if text_input_ids is None:
            text_inputs = self.tokenize(text)
            text_input_ids = text_inputs.input_ids

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = self.text_encoder(text_input_ids.cuda(), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = self.text_encoder(
                text_input_ids.cuda(), attention_mask=attention_mask, output_hidden_states=True
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)
        return prompt_embeds

    def forward(self, texts, **kwargs):
        return self.encode(texts)


if __name__ == '__main__':
    text_encoder = TextEncoder(pretrained_model_name_or_path="/home/yixing/model/stable-diffusion-v1-5-inpainting/text_encoder",
                               tokenizer_name_or_path="/home/yixing/model/stable-diffusion-v1-5-inpainting/tokenizer")
    print(text_encoder)
    input = "a photo of a tiny dog"
    input = input * 20
    text_feature = text_encoder(input)
    print(text_feature.shape)
    breakpoint()