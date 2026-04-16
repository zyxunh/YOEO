# Author: zhuyixing zhuyixing@bytedance.com
# Date: 2023-02-22 12:24:18
# LastEditors: zhuyixing zhuyixing@bytedance.com
# Description:
import numpy as np
import torch
import torch.nn.functional as F

def stack_const(array, x):
    x_array = np.ones((*array.shape[:-1], 1), dtype=array.dtype) * x
    return np.concatenate([array, x_array], axis=-1)

def one_hot(array, dim=1, num_classes=None):
    if num_classes is None:
        num_classes = array.max() + 1
    classes = torch.arange(num_classes, device=array.device, dtype=array.dtype)
    if dim == 1:
        for _ in range(array.dim() - dim):
            classes = classes.unsqueeze_(-1)
        array = array[:, None]
        classes = classes[None, :]
        out = (array == classes).to(array)
    else:
        raise NotImplementedError
    return out

def mask_array_gather(array, mask):
    if isinstance(array, (tuple, list)):
        array = [a for a, m in zip(array, mask) if m]
    else:
        array = array[mask]
    return array


def array_gather(array, idxes):
    if isinstance(array, (tuple, list)):
        array = [array[i] for i in idxes]
    else:
        array = array[idxes]
    return array


def random_select_n(array, n, dim=None):
    if dim is not None:
        raise NotImplementedError
    if isinstance(array, (np.ndarray, list, tuple)):
        return array_gather(array, idxes=np.random.permutation(len(array))[:n])


def kn2nk(keys=None, values=None, mapping=None, batch_size=None):
    if mapping is not None:
        values = [mapping[k] for k in keys]
    if batch_size is None:
        batch_size = len(values[0])
    values = [var if var is not None else (None,) * batch_size for var in values]
    data_nk = []
    for values_ziped in zip(*values):
        data_nk.append(dict(zip(keys, values_ziped)))
    return data_nk


if __name__ == '__main__':
    x = torch.zeros(1, 224, 224)
    out = one_hot(x, num_classes=10)
    pass