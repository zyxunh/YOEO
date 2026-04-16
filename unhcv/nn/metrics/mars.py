import PIL
import math
import numpy as np

from unhcv.common.utils import add_prefix_to_keys
from unhcv.projects.diffusion.inpainting.evaluation.evaluation_metric import RemovingMetric
from .utils import BaseMetric


__all__ = ['MARSMetric']


class MARSMetric(BaseMetric):
    key = "mars_score"

    def __init__(self):
        super().__init__()
        self.remove_metric = RemovingMetric(inter_ratio_thres=0.95)

    def __call__(self, image, inpainting_mask, prefix=None):
        """
        image: 255
        inpainting_mask: 255
        """
        key = self.key
        if isinstance(inpainting_mask, np.ndarray):
            inpainting_mask = PIL.Image.fromarray(inpainting_mask)
            image = PIL.Image.fromarray(image)
        score = self.remove_metric.evaluate_on_sample(None, image, inpainting_mask)
        score_dict = dict(bad_case_num=score['bad_case_num'], bad_case_inter_ratio_to_inpainting=score['bad_case_inter_ratio_to_inpainting'])
        if prefix is not None:
            score_dict = add_prefix_to_keys(score_dict, prefix)
        self.mean_cache.update(score_dict)
        return score_dict


if __name__ == '__main__':
    pass
