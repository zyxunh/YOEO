import itertools
import torch

__all__ = ["add_prefix_to_keys", "concat_unified"]


def add_prefix_to_keys(dct, prefix, end=False):
    if end:
        return {k + prefix: v for k, v in dct.items()}
    else:
        return {prefix + k: v for k, v in dct.items()}


def concat_unified(in_list):
    if isinstance(in_list[0], (list, tuple)):
        return list(itertools.chain(*in_list))
    elif isinstance(in_list[0], torch.Tensor):
        return torch.cat(in_list, dim=0)
    else:
        print('in_list is: ', in_list)
        raise TypeError(f"Unsupported type {type(in_list[0])} for concatenation.")
