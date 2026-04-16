from dataclasses import dataclass, field

from diffusers import UNet2DConditionModel
from diffusers.models.resnet import ResnetBlock2D
import torch
from torch import nn

from unhcv.common.types import DataClass

from .utils import MiniBlock


__all__ = ['UnetConfig', 'Unet', 'UpBlockConfig']


@dataclass
class UpBlockConfig(DataClass):
    low_channels: int = None
    high_channels: int = None
    out_channels: int = None
    temb_channels: int = None
    up_scale: int = 2

@dataclass
class UnetConfig(DataClass):
    channels_lt: tuple
    backbone: nn.Module = None
    out_channels: int = None
    up_block_config: UpBlockConfig = field(default_factory=UpBlockConfig)


class UpBlock(nn.Module):
    def __init__(self, config: UpBlockConfig):
        super().__init__()
        self.up = nn.Upsample(scale_factor=config.up_scale, mode='bilinear', align_corners=False)
        out_channels = config.high_channels if config.out_channels is None else config.out_channels
        self.operator = MiniBlock(config.low_channels + config.high_channels, out_channels, kernel_size=1)

    def forward(self, x_low, x_high):
        x_low = self.up(x_low)
        x = torch.cat([x_low, x_high], dim=1)
        x = self.operator(x)
        return x

class Unet(nn.Module):
    def __init__(self, config: UnetConfig):
        """

        Args:
            channels_lt: (32, 64, 128, 256)
        """
        super().__init__()
        channels_lt = config.channels_lt[::-1]
        config.up_block_config.__init__()
        self.up_modules = nn.ModuleList()
        for i in range(len(channels_lt) - 1):
            # self.up_modules.append(ResnetBlock2D(in_channels=channels_lt[i], out_channels=channels_lt[i+1], temb_channels=None, up=True))
            self.up_modules.append(
                UpBlock(config.up_block_config.update(UpBlockConfig(channels_lt[i], channels_lt[i + 1]))))

        if config.backbone is not None:
            self.backbone = config.backbone
        if config.out_channels is not None:
            self.conv_out_on = True
            self.conv_out = nn.Conv2d(channels_lt[-1], config.out_channels, 1)
        else:
            self.conv_out_on = False

    def forward(self, *, x=None, fpn_features=None):
        if x is not None:
            fpn_features = self.backbone(x)
        fpn_features = fpn_features[::-1]
        x = fpn_features[0]
        unet_features = [x]
        for i in range(len(fpn_features) - 1):
            x = self.up_modules[i](x, fpn_features[i+1])
            # x = self.up_modules[i](x, temb=None) + fpn_features[i+1]
            unet_features.append(x)
        if self.conv_out_on:
            x = self.conv_out(x)
            return x

        return unet_features


if __name__ == '__main__':
    from . import ResNet, MiniBlock

    resnet = ResNet("resnet50")
    unet = Unet(UnetConfig((256, 512, 1024, 2048), resnet, 12))
    x = torch.zeros([1, 3, 256, 256], dtype=torch.float)
    out = unet(x)
    breakpoint()
    import torch
    fpn_features = [torch.randn(1, 32, 64, 64), torch.randn(1, 64, 32, 32), torch.randn(1, 128, 16, 16), torch.randn(1, 256, 8, 8)]
    out = unet(fpn_features)
    breakpoint()
    print('Done')