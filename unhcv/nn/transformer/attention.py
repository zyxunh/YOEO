from typing import Optional

import torch
from diffusers.models.attention_processor import SpatialNorm, Attention as diffusers_Attention
from diffusers.utils import deprecate
from einops import einsum, rearrange
import torch.nn.functional as F


__all__ = ["AttentionUtils", 'Attention']

from torch import nn


class AttentionUtils:
    @classmethod
    def get_attn_score(cls, query, key, scale=None, softmax=True):
        if scale is not None:
            query = query * scale
        attn_score = einsum(query, key, '... l c, ... k c -> ... l k')
        if softmax:
            attn_score = F.softmax(attn_score, dim=-1)
        return attn_score

    @classmethod
    def get_output(cls, value, attn_score):
        output =  einsum(attn_score, value, '... l k, ... k c -> ... l c')
        return output

    @classmethod
    def head_to_batch(cls, x, heads):
        x = rearrange(x, "n l (h_num c) -> n h_num l c", h_num=heads)
        return x

    @classmethod
    def batch_to_head(cls, x):
        x = rearrange(x, "n h_num l c -> n l (h_num c)")
        return x


class Attention(diffusers_Attention):
    pass
