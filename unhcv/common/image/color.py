import cv2
import numpy as np
import torch
import torch.nn.functional as F

from unhcv.common.image import unh_interpolate
from PIL import Image
from PIL.ImageFile import ImageFile


from .color_map import COLOR_MAP
from .visual import visual_tensor
# from unhcv.common.image.color import visual_mask, color2gray

def contrast_enhance(img, max_value=1):
    if isinstance(img, np.ndarray):
        img = img.astype(np.float32)
    else:
        img = img.float()
    img_min = img.min(); img_max = img.max()
    img = (img - img_min) / (img_max - img_min + 1e-4)
    return img

def img_norm_back(img, mean=np.array([123.675, 116.280, 103.530], dtype=np.float32), std=np.array([58.395, 57.120, 57.375], dtype=np.float32), to_rgb=False):
    if isinstance(img, torch.Tensor):
        mean = img.new_tensor(mean); std = img.new_tensor(std)
    else:
        mean = np.array(mean); std = np.array(std)
    img = img * std[None, None, :] + mean[None, None, :]
    if to_rgb:
        if isinstance(img, torch.Tensor):
            img = img.flip(2)
        else:
            img = img[:, :, ::-1]
    return img


def gray2color(gray, color_map=None):
    if color_map is None:
        color_map = COLOR_MAP
    if isinstance(gray, torch.Tensor):
        gray = gray.long()
        color_mask = gray.new_zeros((gray.shape[0], gray.shape[1], 3),
                                    dtype=torch.uint8)
        unique_gray = torch.unique(gray)
    else:
        gray = gray.astype(np.int64)
        color_mask = np.zeros((gray.shape[0], gray.shape[1], 3),
                              dtype=np.uint8)
        unique_gray = np.unique(gray)
    unique_gray = unique_gray[unique_gray != 255]

    for i in unique_gray:
        color = color_map[i % len(color_map)]
        color_mask[gray == i] = color_mask.new_tensor(color) if isinstance(
            color_mask, torch.Tensor) else color
    color = [255, 255, 255]
    color_mask[gray == 255] = color_mask.new_tensor(color) if isinstance(
        color_mask, torch.Tensor) else color
    return color_mask


def visual_mask(img, mask, is_matting=False, contrast_enhance_on=True, stack_axis=0, color_map=None, max_num=None):
    if isinstance(mask, torch.Tensor):
        mask = visual_tensor(mask, max_num=max_num, max_value=255, min_value=0)
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))[..., ::-1]
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if mask.ndim == 3:
        mask, tmp = mask.argmax(0) + 1, mask.max(0)
        mask[tmp == 0] = 0
    if mask.shape[:2] != img.shape[:2]:
        if isinstance(mask, np.ndarray):
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR if is_matting else cv2.INTER_NEAREST_EXACT)
        else:
            mask = unh_interpolate(mask[None, None], img.shape[:2], mode='nearest', align_corners=False)[0, 0]
    if is_matting == False:
        color = gray2color(mask, color_map=color_map)
        if isinstance(img, torch.Tensor):
            color = color.float()
            img = img.float()
        else:
            color = color.astype(np.float32)
            img = img.astype(np.float32)
        img_merge = img.copy() if isinstance(img, np.ndarray) else img.clone()
        valid = mask > 0
        img_merge[valid] = img_merge[valid] * 0.5 + color[valid] * 0.5
    else:
        if contrast_enhance_on:
            mask = contrast_enhance(mask)
        mask = mask[:, :, None]
        color = COLOR_MAP[2][None, None]
        if isinstance(mask, torch.Tensor):
            color = mask.new_tensor(color)
        color = color * mask
        img_merge = img * (1 - mask) + color
    if isinstance(img_merge, torch.Tensor):
        img_concat = torch.cat([img_merge, color], dim=stack_axis)
    else:
        img_merge = img_merge.astype(np.uint8)
        color = color.astype(np.uint8)
        img_concat = np.concatenate([img_merge, color], axis=stack_axis)
    return img_merge, color, img_concat



def color2gray(color):
    if isinstance(color, ImageFile):
        color_idxes = np.array(color.getpalette()).reshape(-1, 3)[:, ::-1]
        gray = np.asarray(color)
        return gray, color_idxes

    if isinstance(color, torch.Tensor):
        assert color.dim() == 3
        color_idxes = torch.unique(color.view(-1, 3), dim=0)
        gray = color.new_zeros(color.shape[:2])
        valid = (color_idxes != color_idxes.new_tensor([[255, 255, 255]])).any(1)
        assert (color_idxes[0] == color_idxes.new_tensor([0, 0, 0])).all()
    elif isinstance(color, np.ndarray):
        color_idxes = np.unique(color.reshape(-1, 3), axis=0)
        gray = np.zeros(color.shape[:2], dtype=color.dtype)
        valid = (color_idxes != np.array([[255, 255, 255]])).any(1)
        assert (color_idxes[0] == np.array([0, 0, 0])).all()
    
    # valid = valid & (color_idxes != color_idxes.new_tensor([[0, 0, 0]])).any(1)
    color_idxes = color_idxes[valid]
    for idx, color_idx in enumerate(color_idxes):
        gray[(color == color_idx[None, None]).all(-1)] = idx
    
    return gray, color_idxes


def generate_idx2color_map(num_classes):
    num_sep_per_channel = int(num_classes ** (1 / 3)) + 1  # 19
    separation_per_channel = 256 // num_sep_per_channel

    color_list = []
    for location in range(num_classes):
        num_seq_r = location // num_sep_per_channel ** 2
        num_seq_g = (location % num_sep_per_channel ** 2) // num_sep_per_channel
        num_seq_b = location % num_sep_per_channel
        assert (num_seq_r <= num_sep_per_channel) and (num_seq_g <= num_sep_per_channel) \
               and (num_seq_b <= num_sep_per_channel)

        R = 255 - num_seq_r * separation_per_channel
        G = 255 - num_seq_g * separation_per_channel
        B = 255 - num_seq_b * separation_per_channel
        assert (R < 256) and (G < 256) and (B < 256)
        assert (R >= 0) and (G >= 0) and (B >= 0)
        assert (R, G, B) not in color_list

        color_list.append((R, G, B))
        # print(location, (num_seq_r, num_seq_g, num_seq_b), (R, G, B))

    return color_list

if __name__ == '__main__':
    im = cv2.imread('/home/tiger/workspace/datasets_nas/demo/seg/video/DAVIS/2017/trainval/Annotations/480p/bear/00000.png')
    gray, color_idxes = color2gray(im)
    color = gray2color(gray, color_idxes)