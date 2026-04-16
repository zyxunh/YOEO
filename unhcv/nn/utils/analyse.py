from collections import OrderedDict

from torch.nn import Module
from typing import Dict

import torch

from unhcv.nn.utils import load_checkpoint


# from unhcv.models.nn.utils.analyse import write_model_structure
__all__ = ['analyse_optim_param', 'analyse_model_param', 'filter_param', 'freeze_model', 'print_memory_status', 'unfreeze_model', 'analyse_state_dict_diff', 'monitor_memory', 'analyse_module_child', 'analyse_model_diff', 'get_model_dtype']



def print_memory_status():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def freeze_model(model):
    """
    Freeze the model
    """
    # model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def filter_param(parameters, mode="requires_grad"):
    if mode == "requires_grad":
        return filter(lambda p: p.requires_grad, parameters)
    elif mode == "not_requires_grad":
        return filter(lambda p: not p.requires_grad, parameters)
    else:
        raise ValueError("mode should be requires_grad or not_requires_grad")


def analyse_optim_param(optim, every=False):
    grad_lt = []
    param_shape_lt = []
    for group in optim.param_groups:
        grad_lt.append(dict(sum=0, grad_num=0, values=[], without_grad_num=0))
        param_shape_lt.append(dict(sum=0, values=[], grad=0, without_grad=0))
        for p in group['params']:
            param_shape_lt[-1]["values"].append(p.shape)
            param_shape_lt[-1]["sum"] += p.numel()
            if every:
                grad_lt[-1]["values"].append(p.grad)
            if p.grad is not None:
                grad_lt[-1]["sum"] += (p.grad.data ** 2).sum()
                grad_lt[-1]["grad_num"] += 1
                param_shape_lt[-1]["grad"] += p.numel()
            else:
                grad_lt[-1]["without_grad_num"] += 1
                param_shape_lt[-1]["without_grad"] += p.numel()
    return grad_lt, param_shape_lt

def analyse_model_param(model):
    grad_dict = OrderedDict(sum=0, values=OrderedDict(), true_num=0, false_num=0)
    param_inform_dict = dict(requires_grad=OrderedDict(sum=0, values=OrderedDict(), num=0),
                             not_requires_grad=OrderedDict(sum=0, values=OrderedDict(), num=0))
    for name, p in model.named_parameters():
        if p.requires_grad:
            container = param_inform_dict["requires_grad"]
        else:
            container = param_inform_dict["not_requires_grad"]
        container["values"][name] = p.shape
        container["sum"] += p.numel()
        container["num"] += 1
        if p.requires_grad:
            if p.grad is not None:
                grad_dict["values"][name] = (p.grad.data ** 2).sum()
                grad_dict["sum"] += grad_dict["values"][name]
                grad_dict["true_num"] += 1
            else:
                grad_dict["values"][name] = None
                grad_dict["false_num"] += 1
    return grad_dict, param_inform_dict

def analyse_model_scale(model):
    trained_parameters = []
    trained_parameters_num = 0
    frozen_parameters_num = 0
    frozen_parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trained_parameters_num += param.numel()
            trained_parameters.append(f"{name}: {list(param.shape)}\n")
        else:
            frozen_parameters_num += param.numel()
            frozen_parameters.append(f"{name}: {list(param.shape)}\n")
    return trained_parameters_num, frozen_parameters_num

def cal_para_num(model):
    requires_num = 0
    not_requires_num = 0
    for name, para in model.named_parameters():
        if para.requires_grad:
            requires_num += para.numel()
        else:
            not_requires_num += para.numel()
    return 'requires_grad: {} M, not_requires_grad: {} M'.format(requires_num / 1e6, not_requires_num / 1e6)

def sum_para_value(model):
    sum_value = [0, 0]
    for name, para in model.named_parameters():
        if para.requires_grad:
            sum_value[0] += para.sum()
        else:
            sum_value[1] += para.sum()
    return 'requires_grad: {}, not_requires_grad: {}'.format(sum_value[0], sum_value[1])

def sum_para_grad_value(model):
    sum_value = 0
    for name, para in model.named_parameters():
        if para.requires_grad and para.grad is not None:
            sum_value += para.grad.sum()

    return 'requires_grad\'s grad value: {}'.format(sum_value)

def sum_para_num(paras):
    num = 0 
    for var in paras:
        num += var.numel()
    return num

def get_para(model):
    para_requires_grad = {}
    para_not_requires_grad = {}
    for name, para in model.named_parameters():
        if para.requires_grad:
            para_requires_grad[name] = para
        else:
            para_not_requires_grad[name] = para
    return para_requires_grad, para_not_requires_grad

def write_model_structure(model: torch.nn.Module, file_path):
    with open(file_path, 'w') as f:
        f.write(model.__str__())


def analyse_state_dict_diff(state_dict_0: Dict[str, torch.Tensor], state_dict_1: Dict[str, torch.Tensor]):
    state_dict_set_0 = set(state_dict_0.keys())
    state_dict_set_1 = set(state_dict_1.keys())
    print("diff 1 - 0", state_dict_set_1 - state_dict_set_0)
    print("diff 0 - 1", state_dict_set_0 - state_dict_set_1)
    diff_dict = {}
    for key in state_dict_set_0 & state_dict_set_1:
        if state_dict_0[key].shape != state_dict_1[key].shape:
            print(f"Shape mismatch {key}, {state_dict_0[key].shape}, {state_dict_1[key].shape}")
        else:
            diff = torch.abs(state_dict_0[key] - state_dict_1[key]).max()
            # print(f"Max diff for {key}: {diff.max()}")
            diff_dict[key] = diff
    max_diff = max(diff_dict.values())
    return diff_dict, max_diff


def monitor_memory(state=None, key=None):
    memory = torch.cuda.memory_allocated() / 1024 ** 3
    msg = f"state: {state}, memory: {memory} GB"
    if key is not None:
        msg = key + " " + msg
    print(msg)
    return memory

def analyse_module_child(module):
    for name, child in module.named_children():
        print(f"子模块名称: {name}, 类型: {type(child).__name__}")

def analyse_model_diff(model1, model2):
    def get_state_dict(model):
        if isinstance(model, Module):
            state_dict = model.state_dict()
        elif isinstance(model, str):
            state_dict = load_checkpoint(state_dict=model)
        else:
            state_dict = model
        return state_dict

    model_1_state_dict = get_state_dict(model1)
    model_2_state_dict = get_state_dict(model2)
    diff_dict = {}
    diff_is_zero = []
    for key in model_1_state_dict.keys():
        if key in model_2_state_dict:
            value_1 = model_1_state_dict[key]
            value_2 = model_2_state_dict[key]
            diff = (value_2.to(value_1) - value_1).abs().mean().item()
            if diff == 0:
                diff_is_zero.append(key)
            else:
                diff_dict[key] = diff
    in_1_not_2 = [var for var in model_1_state_dict.keys() if var not in model_2_state_dict]
    in_2_not_1 = [var for var in model_2_state_dict.keys() if var not in model_1_state_dict]

    param_num1 = sum([var.numel() for var in model_1_state_dict.values()])
    param_num2 = sum([var.numel() for var in model_2_state_dict.values()])

    print('\ndiff_is_zero', diff_is_zero)
    print('\ndiff_dict', diff_dict)
    print('\nin_1_not_2', in_1_not_2)
    print('\nin_2_not_1', in_2_not_1)
    print('\nparam_num1', param_num1)
    print('\nparam_num2', param_num2)
    print('\n')
    return diff_dict, in_1_not_2, in_2_not_1

def get_model_dtype(model):
    param = next(iter(model.parameters()))
    return param

if __name__ == '__main__':
    model = torch.nn.Conv2d(1, 1, 1)
    k = analyse_model_diff(model, model)
    breakpoint()
    write_model_structure(model, '/home/tiger/debug.txt')

