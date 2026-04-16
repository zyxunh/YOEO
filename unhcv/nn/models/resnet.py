from typing import Optional

import torch.nn
import torchvision.models as models
from diffusers.models.resnet import ResnetBlock2D
from torch import Tensor, nn
from torchvision.models.resnet import Bottleneck
from transformers.activations import get_activation

from unhcv.common.utils import find_path
from unhcv.core.train import AccelerateTrainer
from unhcv.nn.utils import load_checkpoint, NaiveSequential, analyse_module_channels


class ResBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        conv_shortcut: bool = False,
        groups: int = 8,
        eps: float = 1e-6,
        non_linearity: str = "leaky_relu",
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None

        self.use_in_shortcut = self.in_channels != out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        if self.use_conv_shortcut:
            shortcut = self.conv_shortcut(shortcut)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + shortcut
        x = self.nonlinearity(x)
        return x

class ResNet(torch.nn.Module):
    def __init__(self, model_name, in_channels=None, checkpoint=None, block_mode=False):
        super().__init__()
        self.model = getattr(models, model_name)()
        if in_channels is not None:
            self.model.conv1 = nn.Conv2d(in_channels, self.model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

        del self.model.avgpool, self.model.fc
        if checkpoint is not None:
            load_checkpoint(self.model, checkpoint, mismatch_shape=True, mismatch_resolve_function=lambda x,y,z: None)
        self.stage = -1
        self.block_index = -1
        model = self.model
        if block_mode:
            del self.model
            self.blocks = nn.ModuleList(
                [nn.Sequential(model.conv1, model.bn1, model.relu), nn.Sequential(model.maxpool, model.layer1[0]), *model.layer1[1:], *model.layer2,
                 *model.layer3, *model.layer4])

            self.num_blocks = len(self.blocks)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        fpn = []
        model = self.model
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        fpn.append(x)
        x = model.layer2(x)
        fpn.append(x)
        x = model.layer3(x)
        fpn.append(x)
        x = model.layer4(x)
        fpn.append(x)

        return fpn

    def forward_stage(self, x, stage=None):
        # See note [TorchScript super()]
        model = self.model
        if stage is None:
            self.stage += 1
            stage = self.stage
        if stage == 0:
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
        elif stage == 1:
            x = model.layer2(x)
        elif stage == 2:
            x = model.layer3(x)
        elif stage == 3:
            x = model.layer4(x)
            self.stage = -1
        return x

    def forward_block(self, x):
        self.block_index += 1
        x = self.blocks[self.block_index](x)
        if self.block_index == self.num_blocks - 1:
            self.block_index = -1
        return x

    @property
    def block_end(self):
        return self.block_index == -1


def analyse_resnet_channels(module):
    if isinstance(module, Bottleneck):
        return analyse_module_channels(module.conv3)


if __name__ == "__main__":
    model = ResNet("resnet50", checkpoint=find_path("model/resnet/resnet50-0676ba61.pth"), block_mode=True)
    print(model)
    # channels = analyse_module_channels([var for var in model.blocks], analyse_resnet_channels)
    x = torch.zeros([1, 3, 224, 224], dtype=torch.float)
    # fpn = model(x)
    breakpoint()
    AccelerateTrainer.save_model_information(model, "resnet50_raw", "/home/yixing/train_outputs/model")
    x_ = x
    while(1):
        x_ = model.forward_block(x_)
        if model.block_index == -1:
            break
    breakpoint()
    pass