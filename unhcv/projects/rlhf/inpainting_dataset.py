import random

import cv2
import torch
from PIL import Image

from unhcv.common.image import mask_proportion, gray2color, resize_img, mask2bounding_box, mask_ratio
from unhcv.common.utils import point_to_line_distance
from unhcv.common.utils.global_item import GLOBAL_ITEM
from unhcv.datasets.common_datasets import DatasetWithPreprocess
from typing import Optional
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from unhcv.datasets.lama import get_mask_generator
from unhcv.datasets.segmentation import Entity2Rgb
import numpy as np


class InpaintingDatasetWithMask(DatasetWithPreprocess):
    default_mask_generator_kwargs = dict(irregular_proba=0.5,
                                         irregular_kwargs=dict(max_angle=4, max_len=60,
                                                               max_width=100, min_times=1,
                                                               max_times=10),
                                         box_proba=0.5,
                                         box_kwargs=dict(margin=10, bbox_min_size=50,
                                                         bbox_max_size=200, min_times=1,
                                                         max_times=3),
                                         segm_proba=0., segm_kwargs=None,
                                         squares_proba=0, squares_kwargs=None,
                                         superres_proba=0, superres_kwargs=None,
                                         outpainting_proba=0, outpainting_kwargs=None,
                                         use_mask_step_id=False)
    erode_prob = 0.5
    filter_erode_line = True
    null_label = 0

    def __init__(
            self,
            *,
            data_indexes_path: Optional[str] = None,
            data_root: Optional[str] = None,
            transforms_kwargs=dict(),
            image_keys=("image", "mask"),
            image_modes=None,
            segmentation_dc_id=0,
            iou_filter_thres=0.25,
            collect_keys=None,
            shuffle=False,
            debug=False,
            mask_generator_kwargs={},
            batch_size=None, backend_config={}, parallel_read_num=None,
            data_indexes_filter=None, text_dropout_prob=0, mask_dropout_prob=0,
            default_text=None, name_pair=None, entity_rgb=False,
            inpainting_region_first_for_mask_id=False,
            use_mask_outpainting_region=False,
            shuffle_mask_id=False,
            max_num_of_mask_id=None,
            # edit
            edit_ratio=0, random_mask_ratio_for_edit=0.1, merge_mask_ratio_for_edit=0, edit_times_range=(1, 4),
            minimum_edit_entity_ratio=0.05, edit_rect_ratio=0.5, edit_random_ratio=0.5,
            edit_rect_config=dict(random_expand_ratio=0.5), edit_random_config=dict(max_dilate=20, min_dilate=3),
            sundries_mask=False, sundries_thres=0.9,
            # add mask in inpainting
            num_entity_mask_range=(1,3), entity_mask_random_rescale=(0.7, 1.3), entity_dilate_range=(0, 6),
            remove_ratio_sub=dict(lama=1, merge=0, entity=0), thing_mask_proportion_thres=20, min_connected_ratio=0,
            inpainting_area_ratio=None,
            # for clip
            return_masked_object_image=False, image_mean=0.5, image_std=0.5,
            # for amodal
            amodal_mode=False, amodal_line_keep_ratio=(0.2, 0.95)
    ):
        super().__init__(data_indexes_path=data_indexes_path, data_root=data_root, transforms_kwargs=transforms_kwargs,
                         image_keys=image_keys, image_modes=image_modes, collect_keys=collect_keys, shuffle=shuffle,
                         debug=debug, batch_size=batch_size, backend_config=backend_config, parallel_read_num=parallel_read_num,
                         data_indexes_filter=data_indexes_filter, name_pair=name_pair)
        _mask_generator_kwargs = self.default_mask_generator_kwargs.copy()
        _mask_generator_kwargs.update(mask_generator_kwargs)
        self.use_mask_step_id = _mask_generator_kwargs.get("use_mask_step_id", False)
        self.lama_mask_generator = get_mask_generator(None, _mask_generator_kwargs)
        self.segmentation_dc_id = segmentation_dc_id
        assert segmentation_dc_id == 0
        self.iou_filter_thres = iou_filter_thres
        self.text_dropout_prob = text_dropout_prob
        self.mask_dropout_prob = mask_dropout_prob
        self.default_text = default_text
        self.entity_rgb = entity_rgb
        self.entity2rgb = Entity2Rgb(grid_num=16)
        self.inpainting_region_first_for_mask_id = inpainting_region_first_for_mask_id
        self.use_mask_outpainting_region = use_mask_outpainting_region
        self.shuffle_mask_id = shuffle_mask_id
        self.max_num_of_mask_id = max_num_of_mask_id

        self.minimum_edit_entity_ratio = minimum_edit_entity_ratio
        self.edit_rect_ratio = edit_rect_ratio
        self.edit_random_ratio = edit_random_ratio
        self.edit_rect_config = edit_rect_config
        self.edit_random_config = edit_random_config
        self.edit_ratio = edit_ratio
        self.remove_ratio = 1 - edit_ratio
        self.edit_times_range = edit_times_range
        self.random_mask_ratio_for_edit = random_mask_ratio_for_edit
        self.merge_mask_ratio_for_edit = merge_mask_ratio_for_edit
        self.sundries_mask = sundries_mask
        self.sundries_thres = sundries_thres

        assert self.edit_ratio + self.remove_ratio == 1

        # add mask in inpainting mask
        if isinstance(num_entity_mask_range, int):
            num_entity_mask_range = (num_entity_mask_range, num_entity_mask_range + 1)
        self.num_entity_mask_range = num_entity_mask_range
        self.entity_mask_random_rescale = entity_mask_random_rescale
        self.entity_dilate_range = entity_dilate_range
        self.remove_ratio_sub = remove_ratio_sub
        self.thing_mask_proportion_thres = thing_mask_proportion_thres
        self.min_connected_ratio = min_connected_ratio
        self.inpainting_area_ratio = inpainting_area_ratio

        # for clip
        self.return_masked_object_image = return_masked_object_image
        self.image_transforms = Compose([ToTensor(), Normalize(mean=image_mean, std=image_std)])
        # for amodal
        self.amodal_mode = amodal_mode
        self.amodal_line_keep_ratio = amodal_line_keep_ratio

    def init_transforms(self, transforms_kwargs):
        super().init_transforms(transforms_kwargs)

    def preprocess_mask(self, data):
        # if isinstance(data['mask'], Image.Image):
        #     data['mask'] = np.array(data['mask'])
        # mask_score = np.array(data['thing_score_mask'])
        # data['mask'][mask_score < 80] = 0
        return data

    def preprocess_thing_mask(self, data):
        if isinstance(data['mask'], Image.Image):
            mask = np.array(data['mask'])
        else:
            mask = data['mask']
        mask_score = np.array(data['thing_score_mask'])
        entity_proportion_mask = np.array(data['entity_proportion_mask'])
        mask = mask.copy()
        mask[mask_score < 100] = 0
        mask[entity_proportion_mask < self.thing_mask_proportion_thres] = 0
        return mask

    def mask2rgb(self, mask):
        mask_rgb = np.zeros([*mask.shape[:2], 3], dtype=np.float32)
        mask_ids = np.unique(mask)
        mask_ids = mask_ids[mask_ids != 0]
        height, width = mask.shape
        for mask_id in mask_ids:
            mask_i = mask_id == mask
            ys, xs = np.where(mask_i)
            yc, xc = ys.mean() / height, xs.mean() / width
            mask_rgb[mask_i] = self.entity2rgb.get_color(yc, xc)
        return mask_rgb

    def preprocess_for_edit(self, data, segmentation_mask, segmentation_mask_ids):
        data_mask = self.preprocess_thing_mask(data)
        data_mask_indexes = np.unique(data_mask)
        data_mask_indexes = data_mask_indexes[data_mask_indexes != 0]
        num_entity_mask = np.random.randint(*self.num_entity_mask_range)
        data_mask_indexes = data_mask_indexes[:num_entity_mask]
        inpainting_mask = np.zeros_like(data["mask"], dtype=np.uint8)
        for index in data_mask_indexes:
            mask_index = (data["mask"] == index).astype(np.uint8)
            entity_dilate_iterations = np.random.randint(*self.entity_dilate_range)
            if entity_dilate_iterations:
                mask_index = cv2.dilate(mask_index, kernel=np.ones([3, 3], dtype=np.uint8),
                                        iterations=entity_dilate_iterations)
            inpainting_mask[mask_index == 1] = 1
        return inpainting_mask

    def mask_entity_augment(self, mask_entity):
        mask_entity = mask_entity.astype(np.uint8)
        mask_entity = cv2.dilate(mask_entity, kernel=np.ones([3, 3], dtype=np.uint8), iterations=1)
        if np.random.rand() < self.edit_rect_ratio:
            inpainting_mask = np.zeros_like(mask_entity, dtype=np.uint8)
            rect = cv2.boundingRect(mask_entity)
            x0, y0, x1, y1 = rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]
            random_expand_ratio = self.edit_rect_config.get("random_expand_ratio")
            x0_min = x0 - rect[2] * random_expand_ratio
            x0_max = x0
            y0_min = y0 - rect[3] * random_expand_ratio
            y0_max = y0
            x1_max = x1 + rect[2] * random_expand_ratio
            x1_min = x1
            y1_max = y1 + rect[3] * random_expand_ratio
            y1_min = y1
            x0_min = max(x0_min, 0)
            y0_min = max(y0_min, 0)
            x1_max = min(x1_max, mask_entity.shape[1])
            y1_max = min(y1_max, mask_entity.shape[0])
            x0 = int(np.random.uniform(x0_min, x0_max) + 0.5)
            y0 = int(np.random.uniform(y0_min, y0_max) + 0.5)
            x1 = int(np.random.uniform(x1_min, x1_max) + 0.5)
            y1 = int(np.random.uniform(y1_min, y1_max) + 0.5)
            inpainting_mask[y0:y1, x0:x1] = 1
        else:
            inpainting_mask = mask_entity
            max_dilate = self.edit_random_config.get("max_dilate")
            min_dilate = self.edit_random_config.get("min_dilate")
            dilate = np.random.randint(min_dilate, max_dilate)
            inpainting_mask = cv2.dilate(inpainting_mask, kernel=np.ones([3, 3], dtype=np.uint8), iterations=dilate)
        return inpainting_mask

    def preprocess_for_remove(self, data, inpainting_mask, segmentation_mask, segmentation_mask_ids):
        if self.iou_filter_thres != 1:
            inpainting_mask_bool = inpainting_mask > 0
            segmentation_mask_ids = np.unique(segmentation_mask[inpainting_mask_bool])
            segmentation_mask_ids = segmentation_mask_ids[segmentation_mask_ids != self.segmentation_dc_id]
            segmentation_mask_dc = segmentation_mask == self.segmentation_dc_id

            if self.debug:
                data["inpainting_mask_raw"] = inpainting_mask.copy()
            mask_erodes = np.zeros_like(segmentation_mask, dtype=np.uint8)
            for segmentation_mask_id in segmentation_mask_ids:
                _mask = mask = segmentation_mask == segmentation_mask_id
                mask_sum = mask.sum()
                if mask_sum == 0:
                    continue
                inpainting_mask_on_segmentation = inpainting_mask_bool & mask
                threshold_num = self.iou_filter_thres * mask_sum
                inpainting_mask_on_segmentation_num = inpainting_mask_on_segmentation.sum()
                if inpainting_mask_on_segmentation_num > threshold_num:
                    if self.use_mask_step_id:
                        mask_id_num = np.bincount(inpainting_mask[mask])
                        mask_ids = np.arange(len(mask_id_num))
                        mask_id_num = mask_id_num[1:];
                        mask_ids = mask_ids[1:]
                        valid = mask_id_num != 0
                        mask_id_num = mask_id_num[valid];
                        mask_ids = mask_ids[valid]
                        random_indexes = np.random.permutation(len(mask_id_num))
                        mask_id_num = mask_id_num[random_indexes]
                        mask_ids = mask_ids[random_indexes]
                        mask_id_num_cumsum = np.cumsum(mask_id_num)
                        for i_mask_index in range(len(mask_id_num_cumsum)):
                            if mask_id_num[i_mask_index] == 0:
                                continue
                            if mask_id_num_cumsum[i_mask_index] - threshold_num < 0:
                                inpainting_mask_on_segmentation[
                                    inpainting_mask_on_segmentation & (inpainting_mask == mask_ids[i_mask_index])] = 0
                            else:
                                break
                        mask = inpainting_mask_on_segmentation
                        assert inpainting_mask_on_segmentation_num - inpainting_mask_on_segmentation.sum() < threshold_num

                    if np.random.rand() < self.erode_prob:
                        mask = mask.astype(np.uint8)
                        mask_erode = cv2.erode(mask, kernel=np.ones([3, 3], dtype=np.uint8), iterations=2)
                        mask_erodes[mask != mask_erode] = 1
                        mask = mask_erode
                    inpainting_mask[mask == 1] = 0
            if self.filter_erode_line:
                inpainting_mask_open = inpainting_mask.astype(np.uint8)
                inpainting_mask_open = cv2.erode(inpainting_mask_open, kernel=np.ones([3, 3], dtype=np.uint8),
                                                 iterations=3)
                inpainting_mask_open = cv2.dilate(inpainting_mask_open, kernel=np.ones([3, 3], dtype=np.uint8),
                                                  iterations=4)
                inpainting_mask[(inpainting_mask_open == 0) & ((mask_erodes == 1) | segmentation_mask_dc)] = 0
        if not self.debug:
            inpainting_mask = (inpainting_mask > 0).astype(np.uint8)
        return inpainting_mask

    def delete_part_for_amodal(self, mask, another_mask=None):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros_like(mask)
        inpainting_mask = np.zeros_like(mask, dtype=np.uint8)
        xys = np.stack([xs, ys], 1)
        theta = random.uniform(0, 2 * np.pi)
        direction = np.stack([np.cos(theta), np.sin(theta)], 0)
        distance = point_to_line_distance(xys, line_direction=direction)
        distance_sort_indices = np.argsort(distance)
        keep_ratio = random.uniform(*self.amodal_line_keep_ratio)
        keep_num = int(round(len(distance_sort_indices) * keep_ratio))
        delete_indices = distance_sort_indices[keep_num:]
        delete_xys = xys[delete_indices]
        inpainting_mask[delete_xys[:, 1], delete_xys[:, 0]] = 1
        inpainting_mask = cv2.dilate(inpainting_mask, kernel=np.ones([5, 5], dtype=np.uint8), iterations=8)
        inpainting_mask = cv2.erode(inpainting_mask, kernel=np.ones([5, 5], dtype=np.uint8), iterations=5)
        return inpainting_mask

    def augment_for_amodal(self, inpainting_mask):
        pass

    def preprocess_for_amodal(self, data, inpainting_mask, segmentation_mask, segmentation_mask_ids):
        inpainting_mask_max_id = inpainting_mask.max() + 1
        segmentation_mask_num = np.bincount(segmentation_mask.reshape(-1))
        segmentation_mask_num_real = segmentation_mask_num[segmentation_mask_ids]
        segmentation_mask_num_real_sort_indices = segmentation_mask_num_real.argsort()[::-1]
        segmentation_mask_ids = segmentation_mask_ids[segmentation_mask_num_real_sort_indices]
        for segmentation_mask_id in segmentation_mask_ids:
            mask = segmentation_mask == segmentation_mask_id
            inpainting_mask_tmp = self.delete_part_for_amodal(mask)
            inpainting_mask[inpainting_mask_tmp > 0] = inpainting_mask_max_id
            inpainting_mask_max_id += 1
        return inpainting_mask

    def filter_small_area(self, mask):
        if self.min_connected_ratio == 0:
            return mask
        # connectivity=8 表示8邻域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # stats: 每一行是 [x, y, w, h, area]
        # 第 0 行是背景
        areas = [var[cv2.CC_STAT_AREA] for var in stats]
        if len(areas) <= 1:
            return mask
        max_area = max(areas[1:])
        area_thres = max_area * self.min_connected_ratio
        for i in range(1, num_labels):  # 从 1 开始，跳过背景
            area = areas[i]
            if area < area_thres:
                mask[labels == i] = 0  # 保留这个连通域
        return mask

    def add_entity_mask(self, inpainting_mask, data):
        if inpainting_mask is None:
            inpainting_mask = np.zeros(data['mask'].shape, dtype=np.uint8)
        num_entity_mask = np.random.randint(*self.num_entity_mask_range)
        inpainting_mask_index_now = inpainting_mask.max() + 1
        if self.inpainting_area_ratio is not None:
            inpainting_area_ratio = random.uniform(*self.inpainting_area_ratio)
        else:
            inpainting_area_ratio = None
        if num_entity_mask > 0:
            data_mask = self.preprocess_thing_mask(data)
            data_mask_indexes = np.unique(data_mask)
            data_mask_indexes = data_mask_indexes[data_mask_indexes != 0]

            random_data = super().read_with_index(np.random.randint(len(self)))
            random_data = self.preprocess_mask(random_data)
            random_data_mask = self.preprocess_thing_mask(random_data)
            random_data_mask_indexes = np.unique(random_data_mask)
            random_data_mask_indexes = random_data_mask_indexes[random_data_mask_indexes != 0]
            random_data_mask_indexes = random_data_mask_indexes + 1000

            concat_index = np.concatenate([data_mask_indexes, random_data_mask_indexes])
            np.random.shuffle(concat_index)
            if inpainting_area_ratio is None:
                concat_index = concat_index[:num_entity_mask]
            for index in concat_index:
                if index >= 1000:
                    index = index - 1000
                    mask = random_data_mask
                else:
                    mask = data_mask
                mask_index = (mask == index).astype(np.uint8)
                scale = np.random.uniform(*self.entity_mask_random_rescale)
                mask_index = cv2.resize(mask_index, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST_EXACT)
                entity_dilate_iterations = np.random.randint(*self.entity_dilate_range)
                if entity_dilate_iterations:
                    mask_index = cv2.dilate(mask_index, kernel=np.ones([3, 3], dtype=np.uint8), iterations=entity_dilate_iterations)
                mask_index = self.filter_small_area(mask_index)
                x, y, w, h = cv2.boundingRect(mask_index.astype(np.uint8))
                inpainting_mask_h, inpainting_mask_w = inpainting_mask.shape
                margin_w = margin_h = 10
                margin_w = min(max(w - 5, 0), margin_w)
                margin_h = min(max(h - 5, 0), margin_h)
                x0_min = -margin_w
                x0_max = inpainting_mask_w - 1 - w + margin_w
                y0_min = -margin_h
                y0_max = inpainting_mask_h - 1 - h + margin_h
                x0_max = max(x0_max, x0_min)
                y0_max = max(y0_max, y0_min)
                x0 = np.random.randint(x0_min, x0_max + 1)
                y0 = np.random.randint(y0_min, y0_max + 1)
                y0_clamp = np.maximum(y0, 0)
                y1_clamp = np.minimum(y0+h, inpainting_mask_h)
                x0_clamp = np.maximum(x0, 0)
                x1_clamp = np.minimum(x0+w, inpainting_mask_w)
                y0_gap = y0_clamp - y0
                y1_gap = y1_clamp - (y0+h)
                x0_gap = x0_clamp - x0
                x1_gap = x1_clamp - (x0+w)
                # try:
                try:
                    inpainting_mask[y0_clamp:y1_clamp, x0_clamp:x1_clamp][mask_index[y+y0_gap:y+h+y1_gap, x+x0_gap:x+w+x1_gap] > 0] = inpainting_mask_index_now
                except:
                    print(mask_index[y+y0_gap:y+h+y1_gap, x+x0_gap:x+w+x1_gap].shape)
                    print(inpainting_mask[y0_clamp:y1_clamp, x0_clamp:x1_clamp].shape)
                    print(inpainting_mask.shape)
                    print(y0_clamp, y1_clamp, x0_clamp, x1_clamp)
                    raise ValueError
                inpainting_mask_index_now += 1
                if inpainting_area_ratio is not None:
                    if mask_ratio(inpainting_mask > 0, inpainting_mask.shape, sqrt=False) >= inpainting_area_ratio:
                        break
                # except:
                #     print('wrong')
                #     breakpoint()
        else:
            pass
        return inpainting_mask

    def read_with_index(self, index):
        if not isinstance(index, tuple):
            index = (index,)
            index_int_flag = True
        else:
            index_int_flag = False
        datas = super().read_with_index(index)
        for data in datas:
            segmentation_mask_ids = None
            image_hw = (data["image"].height, data["image"].width)
            if "mask" in data:
                segmentation_mask: np.ndarray = np.array(data["mask"])
                if segmentation_mask.ndim == 3:
                    assert segmentation_mask.shape[-1] == 3
                    segmentation_mask = segmentation_mask.astype(np.uint32)
                    segmentation_mask = segmentation_mask[..., 0] * 255 ** 2 + segmentation_mask[..., 1] * 255 + \
                                        segmentation_mask[..., 0]

                    segmentation_mask_ids = np.unique(segmentation_mask)
                    segmentation_mask_ids = segmentation_mask_ids[segmentation_mask_ids != 0]
                    _segmentation_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)

                    for i_segmentation_mask_id, segmentation_mask_id in enumerate(segmentation_mask_ids):
                        _segmentation_mask[segmentation_mask == segmentation_mask_id] = i_segmentation_mask_id + 1

                    segmentation_mask = _segmentation_mask

                if self.entity_rgb:
                    data["mask_rgb"] = mask_rgb = self.mask2rgb(segmentation_mask)
                data["mask"] = segmentation_mask.astype(np.int64)
                data = self.preprocess_mask(data)

                if segmentation_mask_ids is None:
                    segmentation_mask_ids = np.unique(segmentation_mask)
                    segmentation_mask_ids = segmentation_mask_ids[segmentation_mask_ids != 0]
                rand_for_mode = np.random.rand()
                if rand_for_mode < self.remove_ratio:
                    random_num = np.random.rand()
                    if random_num < self.remove_ratio_sub['lama']:
                        inpainting_mask = self.lama_mask_generator(image_hw)[0].astype(np.uint8)
                    elif random_num < self.remove_ratio_sub['lama'] + self.remove_ratio_sub['merge']:
                        inpainting_mask = self.lama_mask_generator(image_hw)[0].astype(np.uint8)
                        inpainting_mask = self.add_entity_mask(inpainting_mask, data)
                    else:
                        inpainting_mask = self.add_entity_mask(None, data)
                    if self.amodal_mode:
                        inpainting_mask = self.preprocess_for_amodal(data=data, inpainting_mask=inpainting_mask, segmentation_mask=segmentation_mask, segmentation_mask_ids=segmentation_mask_ids)
                    inpainting_mask = self.preprocess_for_remove(data=data, inpainting_mask=inpainting_mask, segmentation_mask=segmentation_mask, segmentation_mask_ids=segmentation_mask_ids)
                else:
                    inpainting_mask = self.preprocess_for_edit(data=data, segmentation_mask=segmentation_mask, segmentation_mask_ids=segmentation_mask_ids)
            else:
                raise NotImplementedError

            data["inpainting_mask"] = inpainting_mask
            if self.return_masked_object_image:
                if not (inpainting_mask > 0).any():
                    masked_object_image = np.zeros([224, 224, 3], dtype=np.uint8)
                    masked_object_inpainting_mask = np.zeros([224, 224], dtype=np.uint8)
                else:
                    inpainting_mask_origin = resize_img(inpainting_mask, size=data["image_origin"].size, mode='nearest')
                    object_box = mask2bounding_box(inpainting_mask_origin, expand_ratio=0.05, round_box=True)
                    masked_object_image = np.array(data["image_origin"].convert("RGB"))[object_box[1]:object_box[3], object_box[0]:object_box[2]]
                    masked_object_inpainting_mask = inpainting_mask_origin[object_box[1]:object_box[3], object_box[0]:object_box[2]]
                    masked_object_image = resize_img(masked_object_image, size=(224, 224), mode='bicubic')
                    masked_object_inpainting_mask = resize_img(masked_object_inpainting_mask, size=(224, 224), mode='nearest')
                data["masked_object_image"] = torch.from_numpy(masked_object_image).permute(2, 0, 1)
                data["masked_object_inpainting_mask"] = torch.from_numpy(masked_object_inpainting_mask)

            if "text" not in data:
                data["text"] = self.default_text
            if "mask" not in data:
                data["mask"] = np.zeros(image_hw, dtype=np.int64)
            if np.random.rand() < self.text_dropout_prob:
                data["text"] = ""
            if np.random.rand() < self.mask_dropout_prob:
                data["inpainting_mask"] = np.ones_like(inpainting_mask)

            if self.inpainting_region_first_for_mask_id or self.use_mask_outpainting_region:
                if self.inpainting_region_first_for_mask_id:
                    mask_inpainting_region_ids = np.unique(data["mask"][data["inpainting_mask"] > 0])
                    if self.shuffle_mask_id:
                        np.random.shuffle(mask_inpainting_region_ids)
                    mask_ids = mask_inpainting_region_ids
                    if self.use_mask_outpainting_region:
                        mask_outpainting_region_ids = np.unique(data["mask"][data["inpainting_mask"] == 0])
                        mask_outpainting_region_ids_valid = (
                                    mask_inpainting_region_ids[:, None] != mask_outpainting_region_ids[None]).all(0)
                        mask_outpainting_region_ids = mask_outpainting_region_ids[mask_outpainting_region_ids_valid]
                        if self.shuffle_mask_id:
                            np.random.shuffle(mask_outpainting_region_ids)
                        mask_ids = np.concatenate([mask_ids, mask_outpainting_region_ids])
                else:
                    raise NotImplementedError
            if self.max_num_of_mask_id is not None and self.shuffle_mask_id:
                mask_ids = np.unique(data["mask"])
                data["mask"] = self.ids2sorted_mask(data["mask"], mask_ids, max_num_of_mask_id=self.max_num_of_mask_id,
                                                    random_shuffle=self.shuffle_mask_id)

            if self.sundries_mask:
                data["sundries_mask"] = self.produce_sundries_mask(data["inpainting_mask"], data["mask"])

            while not (data["inpainting_mask"] > 0).any():
                index_ = random.randint(0, len(self) - 1)
                data_ = self.read_with_index(index_)
                data.update(data_)

        if index_int_flag:
            datas = datas[0]
        return datas

    def produce_sundries_mask(self, inpainting_mask, mask):
        mask_on_inpainting = mask * (inpainting_mask > 0)
        mask_on_inpainting_ids = np.unique(mask_on_inpainting)
        mask_on_inpainting_ids = mask_on_inpainting_ids[mask_on_inpainting_ids != self.null_label]
        sundries_mask = np.zeros_like(mask)
        inpainting_mask_bool = (inpainting_mask > 0)
        for mask_on_inpainting_id in mask_on_inpainting_ids:
            mask_i = mask == mask_on_inpainting_id
            mask_i_on_inpainting = inpainting_mask_bool & mask_i

            iou = mask_i_on_inpainting.sum() / max(mask_i.sum(), 1)
            if iou > self.sundries_thres:
                if iou < 1:
                    mask_i_dilate = cv2.dilate(mask_i.astype(np.uint8), kernel=np.ones([3, 3], dtype=np.uint8),
                                               iterations=2) == 1
                    inpainting_mask[mask_i_dilate] = 1
                sundries_mask[mask_i] = 1

        return sundries_mask

    def ids2sorted_mask(self, mask, ids, random_shuffle=False, max_num_of_mask_id=None):
        ids = ids[ids != 0]
        if max_num_of_mask_id is not None:
            ids = ids[:max_num_of_mask_id]

        if random_shuffle:
            assert max_num_of_mask_id is not None
            new_ids = np.random.permutation(max_num_of_mask_id) + 1
            new_ids = new_ids[:len(ids)]
        else:
            new_ids = np.arange(len(ids)) + 1
        new_mask = np.zeros_like(mask)
        for i_id, id in zip(new_ids, ids):
            new_mask[mask == id] = i_id
        return new_mask

    # def __getitem__(self, index):
    #     data = self.read_with_index(index)
    #     return data

    def postprocess(self, data):
        data['image'] = self.image_transforms(data['image'])
        data['mask'] = torch.from_numpy(data['mask'])
        data['inpainting_mask'] = torch.from_numpy(data['inpainting_mask']).to(torch.uint8)
        if 'mask_rgb' in data:
            data['mask_rgb'] = torch.from_numpy(data['mask_rgb']).permute(2, 0, 1)
        return super().postprocess(data)

    def preprocess(self, data):
        data_ = super().preprocess(data)
        if self.return_masked_object_image:
            data_['image_origin'] = data['image']
        return data_

    def __iter__(self):
        return super().__iter__()


if __name__ == "__main__":
    pass