import os
import random

import cv2
import torch
from PIL import Image

from unhcv.common import visual_mask, write_im
from unhcv.common.image import mask_proportion, gray2color, resize_img, mask2bounding_box, visual_tensor, \
    concat_differ_size
from unhcv.common.utils import attach_home_root
from unhcv.common.utils.global_item import GLOBAL_ITEM
from unhcv.datasets.common_datasets import DatasetWithPreprocess
from typing import Optional, List, Union
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from unhcv.datasets.lama import get_mask_generator
from unhcv.datasets.segmentation import Entity2Rgb
import numpy as np


__all__ = ["ErasureDataset"]


class ErasureDataset(DatasetWithPreprocess):

    def __init__(
            self,
            *,
            data_indexes_path: Optional[str] = None,
            data_root: Optional[Union[str, List[str]]] = None,
            transforms_kwargs=dict(interpolations=("bicubic", "nearest-exact", "nearest-exact", "nearest-exact")),
            image_keys=("image", "mask", "thing_score_mask", "entity_proportion_mask"),
            image_modes=("RGB", "L", "L", "L"),
            collect_keys=None,
            shuffle=False,
            debug=False,
            mask_generator_kwargs={},
            batch_size=None, backend_config={}, parallel_read_num=None,
            data_indexes_filter=None, text_dropout_prob=0, mask_dropout_prob=0,
            default_text="", name_pair=None,
            ####
            image_mean=0, image_std=1,
            ####
            erasure_num=(1, 5), dilate=(3, 8), max_erasure_ratio=0.8

    ):
        super().__init__(data_indexes_path=data_indexes_path, data_root=data_root, transforms_kwargs=transforms_kwargs,
                         image_keys=image_keys, image_modes=image_modes, collect_keys=collect_keys, shuffle=shuffle,
                         debug=debug, batch_size=batch_size, backend_config=backend_config, parallel_read_num=parallel_read_num,
                         data_indexes_filter=data_indexes_filter, name_pair=name_pair)
        self.image_transforms = Compose([ToTensor(), Normalize(mean=image_mean, std=image_std)])
        if isinstance(erasure_num, int):
            erasure_num = (erasure_num, erasure_num)
        self.erasure_num = erasure_num
        self.dilate = dilate
        self.default_text = default_text
        self.max_erasure_ratio = max_erasure_ratio ** 2
        self.max_try_num = 10

    def init_transforms(self, transforms_kwargs):
        super().init_transforms(transforms_kwargs)

    def preprocess_thing_mask(self, data):
        if isinstance(data['mask'], Image.Image):
            mask = np.array(data['mask'])
        else:
            mask = data['mask']
        mask_score = np.array(data['thing_score_mask'])
        entity_proportion_mask = np.array(data['entity_proportion_mask'])
        mask = mask.copy()
        mask[mask_score < 100] = 0
        mask[entity_proportion_mask < 20] = 0
        return mask

    def read_with_index(self, index):
        if not isinstance(index, tuple):
            index = (index,)
            index_int_flag = True
        else:
            index_int_flag = False
        datas = super().read_with_index(index)
        for data in datas:
            segmentation_mask: np.ndarray = np.array(data["mask"])
            if segmentation_mask.ndim == 3:
                raise ValueError

            segmentation_mask_thing = self.preprocess_thing_mask(data)

            segmentation_mask_thing_ids = np.unique(segmentation_mask_thing)
            segmentation_mask_thing_ids = segmentation_mask_thing_ids[segmentation_mask_thing_ids != 0]
            segmentation_mask_thing_num = np.bincount(segmentation_mask_thing.reshape(-1))
            erasure_num = random.randint(*self.erasure_num)
            segmentation_mask_thing_ids = segmentation_mask_thing_ids[:self.max_try_num]

            inpainting_mask = np.zeros_like(segmentation_mask)
            image_area = inpainting_mask.shape[0] * inpainting_mask.shape[1]
            num_picked = 0
            for segmentation_mask_thing_id in segmentation_mask_thing_ids:
                mask = (segmentation_mask == segmentation_mask_thing_id).astype(np.uint8)
                dilate = random.randint(*self.dilate)
                mask = cv2.dilate(mask, kernel=np.ones([3, 3], dtype=np.uint8), iterations=dilate)
                mask = mask > 0
                # inpainting_mask[mask] = 1
                if (inpainting_mask.sum() + mask.sum()) / image_area > self.max_erasure_ratio:
                    pass
                else:
                    inpainting_mask[mask] = 1
                    num_picked += 1
                if num_picked >= erasure_num:
                    break

            data["inpainting_mask"] = inpainting_mask
            data["mask"] = segmentation_mask_thing

            if "text" not in data:
                data["text"] = self.default_text

            while not (inpainting_mask > 0).any():
                index_ = random.randint(0, len(self) - 1)
                data_ = self.read_with_index(index_)
                inpainting_mask = data_["inpainting_mask"]
                data.update(data_)

        if index_int_flag:
            datas = datas[0]
        return datas

    def postprocess(self, data):
        data['image'] = self.image_transforms(data['image'])
        data['mask'] = torch.from_numpy(data['mask'])
        data['inpainting_mask'] = torch.from_numpy(data['inpainting_mask']).to(torch.uint8)
        return super().postprocess(data)

    def preprocess(self, data):
        data_ = super().preprocess(data)
        return data_

    def __iter__(self):
        return super().__iter__()


if __name__ == "__main__":
    data_root = ["dataset/open-images-dataset/train/lmdb/055_wh1.333_num10000", "dataset/open-images-dataset/train/lmdb_1025/055_wh1.333_num10000"]
    data_indexes_path = "dataset/open-images-dataset/train/lmdb_1025/055_wh1.333_num10000_catalog.bson"
    dataset = ErasureDataset(data_indexes_path=data_indexes_path, data_root=data_root, batch_size=5)
    show_root = attach_home_root("show/erasure_dataset")
    for i, data in enumerate(dataset):
        image = data["image"]
        mask = data["mask"]
        inpainting_mask = data["inpainting_mask"]
        image_show = visual_tensor(image[None], reverse=True, min_value=0, max_value=1)
        mask_show = visual_mask(image_show, mask.cpu().numpy())[-1]
        inpainting_mask_show = visual_mask(image_show, inpainting_mask.cpu().numpy())[-1]
        shows = [image_show, mask_show, inpainting_mask_show]
        shows = concat_differ_size(shows)
        write_im(os.path.join(show_root, f"{i:05}.jpg"), shows)
        if i == 100:
            break
    pass