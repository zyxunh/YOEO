from typing import Optional

import torch
from torch import nn
from transformers.activations import get_activation


__all__ = ["MiniBlock"]


class MiniBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_size: int = 3,
            bias: bool = True,
            groups: int = 8,
            eps: float = 1e-6,
            non_linearity: str = "leaky_relu",
            norm: str = "group_norm",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        norm_channels = out_channels

        if norm is None:
            self.norm = nn.Identity()
        elif norm == "group_norm":
            self.norm = torch.nn.GroupNorm(num_groups=groups, num_channels=norm_channels, eps=eps, affine=True)
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        self.operator = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
                                  padding=kernel_size // 2, bias=bias)

        if non_linearity is None:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = get_activation(non_linearity)

    def forward(self, x):
        x = self.operator(x)
        x = self.norm(x)
        x = self.nonlinearity(x)
        return x
