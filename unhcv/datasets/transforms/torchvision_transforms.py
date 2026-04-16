from torchvision.transforms import Compose as TCompose, RandomResizedCrop as TRandomResizedCrop, \
    RandomHorizontalFlip as TRandomHorizontalFlip, ToTensor, Normalize, CenterCrop
import torchvision.transforms.functional as F
from functools import wraps
import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
from torchvision.transforms.functional import InterpolationMode

inverse_modes_mapping_str = {
    "nearest": InterpolationMode.NEAREST,
    "nearest-exact": InterpolationMode.NEAREST_EXACT,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
    "bicubic": InterpolationMode.BICUBIC,
}

def _interpolation_modes_from_str(mode_name: str) -> InterpolationMode:
    # NEAREST = "nearest"
    # NEAREST_EXACT = "nearest-exact"
    # BILINEAR = "bilinear"
    # BICUBIC = "bicubic"
    # # For PIL compatibility
    # BOX = "box"
    # HAMMING = "hamming"
    # LANCZOS = "lanczos"
    return inverse_modes_mapping_str[mode_name]


class RandomHorizontalFlip(TRandomHorizontalFlip):
    def forward(self, imgs):
        if torch.rand(1) < self.p:
            imgs = [F.hflip(img) for img in imgs]
        return imgs


class RandomResizedCrop(TRandomResizedCrop):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BILINEAR,
                 antialias: Optional[Union[str, bool]] = "warn", interpolations=None, random_on=True):
        super().__init__(size, scale, ratio, interpolation, antialias)
        self.interpolations = interpolations
        self.random_on = random_on

    def forward(self, imgs, interpolations=None):
        assert interpolations is not None
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        new_imgs = []
        for i_img, img in enumerate(imgs):
            interpolation = interpolations[i_img] if interpolations is not None else self.interpolation
            if isinstance(interpolation, str):
                interpolation = _interpolation_modes_from_str(interpolation)
            if self.random_on:
                img = F.resized_crop(img, i, j, h, w, self.size, interpolation, antialias=self.antialias)
            else:
                img = F.resize(img, self.size, interpolation, antialias=self.antialias)
            new_imgs.append(img)
        return new_imgs


class RandomResizedWHCrop(RandomResizedCrop):
    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        _, height, width = F.get_dimensions(img)
        ratio_random = torch.empty(1).uniform_(ratio[0], ratio[1]).item()
        scale_random = torch.empty(1).uniform_(scale[0], scale[1]).item()
        if ratio_random > width / height:
            w = scale_random * width
            h = w / ratio_random
        else:
            h = scale_random * height
            w = h * ratio_random
        w, h = int(round(w)), int(round(h))
        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()
        return i, j, h, w

def SimpleWrap(cls):
    @wraps(cls.forward)
    def forward(self, imgs):
        new_imgs = []
        for img in imgs:
            new_imgs.append(super(self).forward(img))
        return new_imgs
    cls.forward = forward
    return cls

class MultiInput:
    def forward(self, imgs, **kwargs):
        new_imgs = []
        for img in imgs:
            new_imgs.append(super().forward(img))
        return new_imgs

# @SimpleWrap
# class ToTensor(ToTensor):
#     pass

# @SimpleWrap
class CenterCrop(MultiInput, CenterCrop): pass

class Compose(TCompose):
    pass

if __name__ == "__main__":
    k = CenterCrop(128)
    x = [torch.randn(1, 3, 100, 100)] * 2
    y = k(x)
    pass
