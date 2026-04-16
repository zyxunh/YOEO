from typing import Optional

import numpy as np
import torch.nn
import torchvision.models as models
from torch import Tensor, nn
from torchvision.models import VGG as TorchVGG
from transformers.activations import get_activation

from unhcv.common.utils import find_path
from unhcv.core.train import AccelerateTrainer
from unhcv.nn.utils import load_checkpoint, NaiveSequential, analyse_module_channels


__all__ = ["VGG"]


class VGG(torch.nn.Module):
    def __init__(self, model_name, in_channels=None, checkpoint=None, block_mode=False, stage_interval=(4,)):
        super().__init__()
        stage_interval = (0, ) + stage_interval
        self.stage_interval = stage_interval
        self.stage_interval_cumsum = np.cumsum(stage_interval)
        self.model: TorchVGG = getattr(models, model_name)()
        if in_channels is not None:
            conv1: nn.Conv2d = self.model.features[0]
            self.model.features[0] = nn.Conv2d(in_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                                               stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)

        del self.model.avgpool, self.model.classifier
        if checkpoint is not None:
            load_checkpoint(self.model, checkpoint, mismatch_shape=True, mismatch_resolve_function=lambda x,y,z: None)

        self.model.features = self.model.features[:self.stage_interval_cumsum[-1]]
        self.model_stages = nn.ModuleList()
        for i in range(len(self.stage_interval_cumsum) - 1):
            self.model_stages.append(self.model.features[self.stage_interval_cumsum[i]:self.stage_interval_cumsum[i+1]])
        del self.model

        self.stage = -1
        self.block_index = -1

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.features(x)
        return x

    def forward_stage(self, x, stage=None):
        if stage is None:
            self.stage += 1
            stage = self.stage
        out = self.model_stages[stage](x)
        return out

    @property
    def block_end(self):
        return self.block_index == -1


def analyse_resnet_channels(module):
    if isinstance(module, Bottleneck):
        return analyse_module_channels(module.conv3)


if __name__ == "__main__":
    model = VGG("vgg16", stage_interval=(16, ))
    print(model)
    # channels = analyse_module_channels([var for var in model.blocks], analyse_resnet_channels)
    x = torch.zeros([1, 3, 224, 224], dtype=torch.float)
    # fpn = model(x)
    out = model.forward_stage(x, 0)
    breakpoint()
    AccelerateTrainer.save_model_information(model, "resnet50_raw", "/home/yixing/train_outputs/model")
    x_ = x
    while(1):
        x_ = model.forward_block(x_)
        if model.block_index == -1:
            break
    breakpoint()
    pass