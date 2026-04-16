import os
import os.path as osp
import random
import numpy as np
import PIL.Image as Image
import torch
from unhcv.common.utils import walk_all_files_with_suffix
from unhcv.common.utils import obj_load
from unhcv.datasets.painter import (Compose, ToTensor,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    RandomApply, ColorJitter)
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from typing import List, Optional, Union


class SegmentationDataset:
    pair_list: List[List[str]] = None

    def __init__(self, image_roots: Union[List, str], image_type_names: Union[List, str] = None, input_size=512,
                 min_random_scale=0.3, shuffle=True):
        if isinstance(image_roots, str):
            image_roots = [image_roots]
        self.image_roots: List[str] = image_roots
        self.image_names = walk_all_files_with_suffix(image_roots[0])
        if image_type_names is None:
            image_type_names = [osp.basename(var) for var in image_roots]
        self.image_type_names = image_type_names
        self.generate_pair_list()
        self.transforms = Compose([
            RandomResizedCrop(input_size, scale=(min_random_scale, 1.0), interpolation=3),  # 3 is bicubic
            RandomApply([
                ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            RandomHorizontalFlip()])
        self.to_tensor = transforms.PILToTensor()
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.shuffle = shuffle

    def generate_pair_list(self):
        pair_list = []
        for image_name in self.image_names:
            pair = [image_name]
            base_name = osp.splitext(osp.basename(image_name))[0]
            for image_root in self.image_roots[1:]:
                pair.append(osp.join(image_root, base_name + ".png"))
            pair_list.append(pair)
        self.pair_list = pair_list
    
    def read_i(self, i):
        pair = self.pair_list[i]
        images = []
        for image_name in pair:
            images.append(Image.open(image_name))
        interpolation1 = 'bicubic'
        interpolation2 = 'nearest'
        images = self.transforms(*images, interpolation1, interpolation2)
        images = [self.normalize(images[0]), self.normalize(images[1])]
        images_dict = dict(zip(self.image_type_names, images))
        return images_dict

    def __len__(self):
        return len(self.image_names)

    def __iter__(self):
        if self.shuffle:
            read_idxes = np.random.permutation(len(self))
        else:
            read_idxes = range(len(self))
        for i in read_idxes:
            yield self.read_i(i)


if __name__ == "__main__":
    from unhcv.datasets.utils.torch_dataset_wrap import wrap_torch_dataset
    from unhcv.projects.diffusion.ldm import multi_patch_noise
    from unhcv.common.image.visual import imwrite_tensor
    segmentation_dataset = wrap_torch_dataset(
        SegmentationDataset(image_roots=["/home/tiger/dataset/ADEChallengeData2016/images/training",
                                         "/home/tiger/dataset/ADEChallengeData2016/annotations_rgb/training"],
                            image_type_names=["image", "segmentation"]))
    dataloader = torch.utils.data.DataLoader(segmentation_dataset, num_workers=1, batch_size=2)
    dataloader_iter = iter(dataloader)
    out = next(dataloader_iter)
    noise = multi_patch_noise(out['segmentation'].shape, (1, 3, 5, 7))
    noise_segmentation = out['segmentation'] * 0.2 + noise * 0.8
    imwrite_tensor("/home/tiger/code/unhcv/show.jpg", noise_segmentation)
    noise = multi_patch_noise(out['segmentation'].shape, (1,))
    noise_segmentation = out['segmentation'] * 0.2 + noise * 0.8
    imwrite_tensor("/home/tiger/code/unhcv/show1.jpg", noise_segmentation)
    breakpoint()
    out = segmentation_dataset.read_i(0)
    segmentation_dataset_iter = iter(segmentation_dataset)
    out = next(segmentation_dataset_iter)
    breakpoint()
    pass
