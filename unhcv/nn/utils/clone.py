import copy

from typing import overload

from torch import nn


__all__ = ["clone_conv", "clone_module"]

from .checkpoint import load_checkpoint


@overload
def clone_conv(conv, *, in_channels=None, out_channels=None, kernel_size=None, stride=None,
               padding=None, dilation=None, groups=None, bias=None, resume_weight=False, zero_init=False) -> nn.Module: ...

def clone_conv(conv: nn.Conv2d, resume_weight=False, zero_init=False, **kwargs) -> nn.Conv2d:
    kwargs_ = dict(in_channels=conv.in_channels, out_channels=conv.out_channels, kernel_size=conv.kernel_size,
                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
                   bias=(conv.bias is not None))
    kwargs_.update(kwargs)
    cloned_conv = nn.Conv2d(**kwargs_)
    if zero_init:
        cloned_conv.weight.data.zero_()
        cloned_conv.bias.data.zero_()
    if resume_weight:
        load_checkpoint(cloned_conv, conv.state_dict(), mismatch_shape=True)
    return cloned_conv

def clone_linear(linear: nn.Linear, resume_weight=False, zero_init=False, **kwargs) -> nn.Linear:
    cloned_linear = nn.Linear(linear.in_features, linear.out_features, linear.bias is not None)
    assert not zero_init, "not implemented"
    return cloned_linear

def clone_layer_norm(layer_norm: nn.LayerNorm, resume_weight=False, zero_init=False, **kwargs) -> nn.LayerNorm:
    config = dict(normalized_shape=layer_norm.normalized_shape, eps=layer_norm.eps, elementwise_affine=layer_norm.elementwise_affine, bias=layer_norm.bias is not None)
    for k in kwargs.keys():
        if k in config:
            config[k] = kwargs[k]
    module = nn.LayerNorm(**config)
    return module


module_map_clone_func = {
    nn.Conv2d: clone_conv,
    nn.Linear: clone_linear,
    nn.LayerNorm: clone_layer_norm,
}


def clone_module(module: nn.Module, *args, deepcopy=False, requires_grad=True, **kwargs) -> nn.Module:
    if deepcopy:
        cloned_module = copy.deepcopy(module)
        if requires_grad:
            cloned_module.requires_grad_(True)
        return cloned_module
    for m, f in module_map_clone_func.items():
        if isinstance(module, m):
            return f(module, *args, **kwargs)
    raise NotImplementedError(module)