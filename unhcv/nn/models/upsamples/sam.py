from typing import Type

from torch import nn

from unhcv.nn.norm import LayerNorm2d


__all__ = ["SAMUpSampler"]


class SAMUpSampler(nn.Module):
    def __init__(self, in_channels, activation: Type[nn.Module] = nn.GELU, out_channels=None):
        super().__init__()
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2, stride=2),
            LayerNorm2d(in_channels // 4),
            activation(),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=2, stride=2),
            activation(),
        )
        if out_channels is not None:
            self.output = nn.Conv2d(in_channels // 8, out_channels, kernel_size=1)
        else:
            self.output = nn.Identity()

    def forward(self, x):
        x = self.output_upscaling(x)
        x = self.output(x)
        return x
        