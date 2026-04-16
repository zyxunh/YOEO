from typing import Dict
import math
import numpy as np
import torch


def round_sigfig(x, sigfig=3):
    if isinstance(x, (int, float)):
        return _round_scalar_sigfig(x, sigfig)
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        return round_to_significant(x, sigfig)
    elif isinstance(x, list):
        return [round_sigfig(v, sigfig) for v in x]
    elif isinstance(x, dict):
        return {k: round_sigfig(v) for k, v in x.items()}
    else:
        raise TypeError("Unsupported type")

def round_to_significant(x: torch.Tensor, n: int):
    # 避免 log10(0)，用 where 屏蔽
    if isinstance(x, torch.Tensor):
        func = torch
        pow = torch.pow
    else:
        func = np
        pow = np.power
    x_is_0 = x == 0
    x[x_is_0] = 1
    order = func.floor(func.log10(func.abs(x)))
    scale = pow(10, order - (n - 1))
    out = func.round(x / scale) * scale
    out = out * ~x_is_0
    return out

def _round_scalar_sigfig(x, sigfig):
    if x == 0:
        return 0.0
    sign = -1 if x < 0 else 1
    x_abs = abs(x)
    magnitude = math.floor(math.log10(x_abs))
    factor = 10 ** (sigfig - 1 - magnitude)
    return sign * round(x_abs * factor) / factor

def human_format_num(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def dict2strs(x_dict: Dict, ndigits=2, return_list=True):
    strs = []
    for key, value in x_dict.items():
        if isinstance(value, float):
            value = _round_scalar_sigfig(value, ndigits)
        strs.append("{}: {}".format(key, value))
    if return_list:
        return strs
    else:
        return ', '.join(strs)

if __name__ == "__main__":
    round_sigfig(torch.tensor(1234567), 3)
    round_sigfig(np.array(1234567, dtype=np.float32), 3)
    breakpoint()
    human_format_num(100000)
    pass