import cv2
import numpy as np


__all__ = ['mask_proportion', 'acquire_images_size', 'mask2bounding_box', 'mask_ratio']

from tqdm import tqdm


def mask_proportion(mask_area, image_shape, sqrt=True):
    if not isinstance(mask_area, (int, float)):
        mask_area = mask_area.sum()
    image_area = image_shape[0] * image_shape[1]
    ratio_length = (mask_area / image_area)
    if sqrt:
        ratio_length = ratio_length ** 0.5
    return ratio_length

mask_ratio = mask_proportion


def acquire_images_size(names):
    import imagesize
    sizes = []
    for name in tqdm(names):
        size_wh = imagesize.get(name)
        sizes.append(size_wh)
    return sizes


def mask2bounding_box(mask, expand_ratio=0, round_box=True):
    x0, y0, w, h = cv2.boundingRect(mask.astype(np.uint8))
    x1 = x0 + w
    y1 = y0 + h
    if expand_ratio != 0:
        mask_h, mask_w = mask.shape[:2]
        w_ratio = w * expand_ratio
        h_ratio = h * expand_ratio
        x0 = x0 - w_ratio; y0 = y0 - h_ratio
        x1 = x1 + w_ratio; y1 = y1 + h_ratio
        box = np.array([x0, y0, x1, y1], dtype=np.float32)
        box[::2] = np.minimum(np.maximum(0, box[::2]), mask_w - 1)
        box[1::2] = np.minimum(np.maximum(0, box[1::2]), mask_h - 1)
    else:
        box = np.array([x0, y0, x1, y1], dtype=np.float32)
    if round_box:
        box = np.round(box).astype(np.int64)
    return box