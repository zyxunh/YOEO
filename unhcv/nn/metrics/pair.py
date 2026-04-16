import argparse
import os.path

import cv2
import torch
import tqdm

from unhcv.common import get_related_path
from unhcv.common.utils import find_path, walk_all_files_with_suffix
from unhcv.nn.utils import to_tensor_unhcv
from .utils import BaseMetric

__all__ = ['PairMetric']



class PairMetric(BaseMetric):
    key = "pair_score"

    def __init__(self, use_mse=True, use_psnr=True, use_ssim=False, only_in_mask=False):
        super().__init__()
        self.use_mse = use_mse
        self.use_psnr = use_psnr
        self.use_ssim = use_ssim
        self.only_in_mask = only_in_mask

    def __call__(self, image1, image2, inpainting_mask=None, prefix=None):
        key = self.key
        score_dict = {}
        if prefix is not None:
            key = f"{prefix}_{key}"
        image1 = to_tensor_unhcv(image1).to(torch.float64)
        image2 = to_tensor_unhcv(image2).to(torch.float64)
        if self.only_in_mask:
            inpainting_mask = to_tensor_unhcv(inpainting_mask)
            inpainting_mask_bool = inpainting_mask > 0
            image1 = image1[inpainting_mask_bool]
            image2 = image2[inpainting_mask_bool]
            if len(image1) == 0:
                return
        diff = image1 - image2
        if self.use_mse:
            mse_diff = (diff ** 2).mean()
            score_dict[f'{key}_mse_diff'] = mse_diff.item()
        if self.use_psnr:
            mse_diff = (diff ** 2).mean().clamp(min=1e-4)
            psnr = 20 * torch.log10(255 / torch.sqrt(mse_diff))
            score_dict[f'{key}_psnr'] = psnr.item()

        self.mean_cache.update(score_dict)
        return score_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_root", type=str)
    parser.add_argument("--gt_root", type=str)
    parser.add_argument("--mask_root", type=str, default=None)
    args = parser.parse_args()

    predict_root = find_path(args.predict_root)
    gt_root = find_path(args.gt_root)
    mask_root = args.mask_root

    if mask_root is not None:
        mask_root = find_path(mask_root)
    pair_metric = PairMetric(only_in_mask=mask_root is not None)
    image_names = walk_all_files_with_suffix(predict_root)
    imags_names_1 = walk_all_files_with_suffix(gt_root)
    assert len(image_names) == len(imags_names_1)

    for image_name in tqdm.tqdm(image_names):
        gt_name = get_related_path(image_name, predict_root, gt_root)
        image = cv2.imread(image_name)
        if not os.path.exists(gt_name):
            gt_name = gt_name.replace(".jpg", ".png")
        gt = cv2.imread(gt_name)
        if mask_root is not None:
            mask_name = get_related_path(image_name, predict_root, mask_root, suffixs=".png")
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
        image = cv2.resize(image, gt.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        score = pair_metric(image, gt, mask)
    pass
    print(pair_metric.result)
