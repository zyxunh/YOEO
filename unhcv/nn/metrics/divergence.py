import math
import numpy as np

from .utils import BaseMetric


__all__ = ['PSNRMetric']


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


class PSNRMetric(BaseMetric):

    def __call__(self, y_true, y_pred, prefix=None):
        score = computePSNR(y_true, y_pred)
        key = "psnr"
        if prefix is not None:
            key = f"{prefix}_{key}"
        self.mean_cache.update({key: score})
        return score
