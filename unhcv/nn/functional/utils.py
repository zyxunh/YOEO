import torch
import torch.nn as nn
import torch.nn.functional as F

from unhcv.common.utils import get_logger

logger = get_logger(__name__)

__all__ = ["clamp_preserving_gradient", "one_hot"]

class GradientPreservingClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min, max):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=min, max=max)  # 正向传播使用普通clamp

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        valid = (x > -10) | (grad_output > 0)
        print(grad_output)
        grad_output = grad_output * valid
        return grad_output, None, None  # 反向传播直接传递梯度，忽略边界

def clamp_preserving_gradient(x, min=None, max=None):
    return GradientPreservingClamp.apply(x, min, max)


def one_hot(x, dim=1, use_unique=False):
    if use_unique:
        indices = torch.unique(x, sorted=True)
        if indices[0] != 0:
            logger.warning(f"indices 0 is not 0 {indices}")
            indices = torch.cat([indices.new_tensor([0]), indices], dim=0)
    else:
        indices = torch.arange(0, x.max() + 1, dtype=x.dtype, device=x.device)
    x = x.unsqueeze(dim)
    x_one_hot = x == indices[None, :, None, None]
    return x_one_hot, indices