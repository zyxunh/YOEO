from einops import rearrange
import torch
import torch.nn as nn


__all__ = ['EinsumLinear']


class EinsumLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, patch=1, einsum_dim=None, equation=None, bias_shape=None):
        super().__init__()
        self.patch = patch
        self.weight = torch.nn.Parameter(torch.randn(patch, out_channels, in_channels))
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(patch, out_channels).view(*bias_shape))
            nn.init.constant_(self.bias, 0)
        else:
            self.bias = None
        self.equation = equation
        self.einsum_dim = einsum_dim


    def forward(self, x, equation=None):
        if equation is None:
            equation = self.equation
        out = torch.einsum(equation, x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out