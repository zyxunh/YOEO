import torch
import torch.nn as nn
from typing import List, Union


class PreNorm(nn.Module):
    pixel_mean: torch.Tensor
    pixel_std: torch.Tensor

    def __init__(
        self,
        pixel_mean: Union[List[float], float] = [0.485, 0.456, 0.406],
        pixel_std: Union[List[float], float] = [0.229, 0.224, 0.225],
    ) -> None:
        super().__init__()
        if isinstance(pixel_mean, (float, int)):
            pixel_mean = [pixel_mean] * 3
            pixel_std = [pixel_std] * 3
        self.register_buffer("pixel_mean",
                             torch.Tensor(pixel_mean).view(1, 3, 1, 1), False)
        self.register_buffer("pixel_std",
                             torch.Tensor(pixel_std).view(1, 3, 1, 1), False)

    def forward(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def inverse(self, x, max=None, min=None):
        std = self.pixel_std; mean = self.pixel_mean
        if max is not None:
            std = std * (max - min)
            mean = mean * (max - min) + min
        return x * std + mean

    def __repr__(self):
        base = super().__repr__()
        base += f"\n  pixel_mean={self.pixel_mean}"
        base += f"\n  pixel_mean={self.pixel_std}"
        return base


class MaskedMerge(nn.Module):
    def __init__(self, pad_value=[-1.7922626, -1.7520971, -1.4802198]) -> None:
        super().__init__()
        self.register_buffer("pad_value",
                             torch.Tensor(pad_value).view(1, 3, 1, 1), False)

    def forward(self, image, mask):
        image = image * mask + self.pad_value * (1 - mask)
        return image


if __name__ == "__main__":
    pass