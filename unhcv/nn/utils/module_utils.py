from collections import defaultdict
from types import MappingProxyType

import torch.nn as nn
from typing import List, Tuple

__all__ = ['walk_all_children', "set_module_to_zero", "NaiveSequential", "analyse_module_channels",
           "build_sequential_module", "ExtraInputModule", "CollectModel", "ReplaceModule"]


def walk_all_children(model: nn.Module, collector=[], reserve_cls=None):
    collector = {}
    for name, module in model.named_modules():
        if isinstance(module, reserve_cls):
            collector[name] = module
    return collector


def set_module_to_zero(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def analyse_module_channels(module, analyse_func=None):
    if analyse_func is not None:
        channels = analyse_func(module)
        if channels is not None:
            return channels
    if isinstance(module, (list, tuple)):
        return [analyse_module_channels(m, analyse_func=analyse_func) for m in module]
    if isinstance(module, (nn.Sequential, nn.ModuleList)):
        i = -1
        while(True):
            channels = analyse_module_channels(module[i], analyse_func=analyse_func)
            if channels is not None:
                return channels
            i -= 1
            if -i > len(module):
                breakpoint()
    if hasattr(module, "out_channels"):
        return module.out_channels


class NaiveSequential:
    def __init__(self, *args):
        self.modules = args

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x


def build_sequential_module(module_cls, common_config=MappingProxyType({}), individual_config=(), num=None):
    if len(individual_config) > 0:
        num = len(individual_config)
    modules = []
    for i in range(num):
        config = individual_config[i] if individual_config is not None else {}
        config = dict(**common_config, **config)
        modules.append(module_cls(**config))
    return modules

class CustomSequential:
    def __init__(self, module_cls):
        super().__init__()

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x


class ExtraInputModule(nn.Module):
    def __init__(self, module, in_keys=()):
        super().__init__()
        self.module = module
        self.in_keys = in_keys

    def forward(self, x, **kwargs):
        _kwargs = {}
        for key in self.in_keys:
            _kwargs[key] = kwargs[key]
        return self.module(x, **_kwargs)


class CollectModel(nn.Module):
    def __init__(self, _base_module=None, **kwargs):
        super().__init__()
        if _base_module is not None:
            self._base_module = _base_module
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, *args, _model_key=None, _func_key=None, **kwargs):
        if _model_key is not None:
            func = getattr(self, _model_key)
        else:
            func = self
            # func = self._base_module
        if _func_key is not None:
            func = getattr(func, _func_key)
        return func(*args, **kwargs)

    @classmethod
    def wrap(cls, model):
        model.forward = cls.forward.__get__(model, model.__class__)
        return model

class ReplaceModule:
    name_now = ""
    def replace(self, module):
        return module

    def need_replace_func(self, module, module_cls):
        return isinstance(module, module_cls)

    def walk_all_children(self, model, module_cls, replace_func=None, need_replace_func=None):
        if need_replace_func is None:
            need_replace_func = self.need_replace_func
        if replace_func is None:
            replace_func = self.replace
        name_before = self.name_now
        for _name, _module in model.named_children():
            self.name_now = name_before + "." +  _name if name_before else name_before + _name
            self.model, self._module = model, _module
            if need_replace_func(_module, module_cls):
                _module = replace_func(_module)
                setattr(model, _name, _module)
            else:
                self.walk_all_children(_module, module_cls, replace_func=replace_func, need_replace_func=need_replace_func)

    @staticmethod
    def setattr_function(module, function_name, function):
        setattr(module, function_name, function.__get__(module, module.__class__))
        return module