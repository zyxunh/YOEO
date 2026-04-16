import math
import numpy as np

from .utils import BaseMetric


__all__ = ['ReMoveMetric']

from ...third_party.remove.remove import RemoveMetric as _RemoveMetric


class ReMoveMetric(BaseMetric):
    key = "remove_score"

    def __init__(self):
        super().__init__()
        self._remove_metric = _RemoveMetric()

    def __call__(self, image, inpainting_mask, prefix=None):
        """
        image: 255
        inpainting_mask: 255
        """
        key = self.key
        if prefix is not None:
            key = f"{prefix}_{key}"
        if not (np.array(inpainting_mask) == 255).any():
            score_dict = {key: 0}
        else:
            score = self._remove_metric.metric(image, inpainting_mask)['remove_score']
            score_dict = {key: score}
            self.mean_cache.update({key: score})
        return score_dict


if __name__ == '__main__':
    pass
