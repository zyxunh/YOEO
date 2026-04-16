import torch
from torch.autograd import Function

__all__ = ["fix_grad_forward", "grad_layer_wrt_loss", "gradient_penalty", "grad_in_mask", "grad_scale"]


class FixGradOperator(Function):
    """
    自定义算子，前向传播为恒等映射，反向传播使用指定的梯度
    """

    @staticmethod
    def forward(ctx, input_tensor, grad_tensor):
        """
        前向传播：返回输入张量的副本

        参数:
            input_tensor: 输入张量
            grad_tensor: 反向传播时使用的特定梯度（可选）
        """
        # 保存输入张量和预定义的梯度到上下文，供反向传播使用
        ctx.save_for_backward(input_tensor, grad_tensor)
        return input_tensor.new_tensor(0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用预定义的梯度而非计算得到的梯度

        参数:
            grad_output: 从后序节点传来的梯度（这里会被忽略）
        """
        input_tensor, grad_tensor = ctx.saved_tensors
        grad_tensor = grad_tensor * grad_output

        return grad_tensor, None

def fix_grad_forward(input_tensor, grad_tensor):
    return FixGradOperator.apply(input_tensor, grad_tensor)


class GradInMaskOperator(Function):

    @staticmethod
    def forward(ctx, input_tensor, mask):
        ctx.save_for_backward(mask)
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        mask,  = ctx.saved_tensors
        grad_output = grad_output * mask

        return grad_output, None

def grad_in_mask(input_tensor, mask):
    return GradInMaskOperator.apply(input_tensor, mask)

class GradScale(Function):

    @staticmethod
    def forward(ctx, input_tensor, scale):
        ctx.scale = scale
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        grad_output = grad_output * scale

        return grad_output, None

def grad_scale(input_tensor, scale, trick=False):
    if trick:
        return input_tensor * scale + input_tensor.detach() * (1 - scale)
    return GradScale.apply(input_tensor, scale)


def grad_layer_wrt_loss(loss, layer):
    return torch.autograd.grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True,
    )[0].detach()


def gradient_penalty(images, output, weight=10):
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    bsz = gradients.shape[0]
    gradients = torch.reshape(gradients, (bsz, -1))
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()