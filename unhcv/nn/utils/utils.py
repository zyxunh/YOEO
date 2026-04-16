import functools
from contextlib import nullcontext

import numpy as np
import torch


__all__ = ["wrap_no_grad", "to_tensor_unhcv"]

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def wrap_no_grad(func):
    """

    Args:
        func:

    Returns:

    """
    @functools.wraps(func)
    def wrapper(*args, requires_grad=False, **kwargs):
        context = nullcontext if requires_grad else torch.no_grad
        with context():
            return func(*args, **kwargs)
    return wrapper

def to_tensor_unhcv(data, fix_dim_order=True):
    if isinstance(data, torch.Tensor):
        pass
    elif isinstance(data, Image.Image):
        data = np.array(data)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    # else:
    #     raise TypeError(f"Unsupported data type: {type(data)}")
    return data