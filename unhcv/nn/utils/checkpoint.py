import os

import torch
import os.path as osp
from typing import Optional, Dict, Union, Any

from unhcv.common.utils import get_logger

logger = get_logger(__name__)

def load_safetensors(file):
    from safetensors import safe_open
    tensors = {}
    with safe_open(file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_checkpoint(model: torch.nn.Module = None, state_dict: Optional[Union[Dict, str]]={}, *, mismatch_shape=False, log_missing_keys=False,
                    replace_name={}, mismatch_resolve_function=None, missing_resolve_function=None, dtype=None):
    """Load checkpoint.

    Args:
        model (nn.Module): The model to be loaded.
        state_dict (dict): Checkpoint state_dict.

    Returns:
        dict: Loaded checkpoint state_dict.
    """
    if isinstance(state_dict, str):
        print(f'load checkpoint from {state_dict}')
        if "::" in state_dict:
            state_dict, model_sub_name = state_dict.split("::")
            model_sub_name = model_sub_name + "."
        else:
            model_sub_name = None
        if osp.splitext(state_dict)[1] == ".safetensors":
            state_dict = load_safetensors(state_dict)
        else:
            state_dict = torch.load(state_dict, map_location='cpu')
        if model_sub_name is not None:
            state_dict_ = {}
            for key in state_dict.keys():
                if key.startswith(model_sub_name):
                    state_dict_[key[len(model_sub_name):]] = state_dict[key]
            state_dict = state_dict_
    if "shadow_params" in state_dict:
        state_dict_ = {}
        for i_key, key in enumerate(model.state_dict().keys()):
            state_dict_[key] = state_dict["shadow_params"][i_key]
        state_dict = state_dict_
    if dtype is not None:
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(dtype)
    if model is None:
        return state_dict
    if len(replace_name):
        _state_dict = {}
        for key, value in state_dict.items():
            for key1, key2 in replace_name.items():
                key = key.replace(key1, key2)
            _state_dict[key] = value
        state_dict = _state_dict
    if mismatch_resolve_function is not None:
        mismatch_shape = True
    if mismatch_shape:
        state_dict_update = state_dict.copy()
        model_dict = model.state_dict()
        for key, state_parameter in state_dict.items():
            if key in model_dict:
                model_parameter = model_dict[key].clone()
                if state_parameter.shape != model_parameter.shape:
                    print(f"Shape mismatch {key}, {model_parameter.shape}, {state_parameter.shape}")
                    if mismatch_resolve_function is not None:
                        state_parameter = mismatch_resolve_function(key, state_parameter, model_parameter)
                    else:
                        assert state_parameter.dim() == model_parameter.dim()
                        _state_parameter = model_parameter
                        if state_parameter.dim() == 4:
                            _state_parameter[
                                : min(state_parameter.size(0), model_parameter.size(0)),
                                : min(state_parameter.size(1), model_parameter.size(1)),
                            ] = state_parameter[
                                : min(state_parameter.size(0), model_parameter.size(0)),
                                : min(state_parameter.size(1), model_parameter.size(1)),
                            ]
                        elif state_parameter.dim() == 1:
                            _state_parameter[key][: min(state_parameter.size(0), model_parameter.size(0))] = state_parameter[
                                : min(state_parameter.size(0), model_parameter.size(0))
                            ]
                        else:
                            raise NotImplementedError('other dim is not implemented')
                        state_parameter = _state_parameter
                    if state_parameter is not None:
                        state_dict_update[key] = state_parameter
                    else:
                        del state_dict_update[key]

        state_dict = state_dict_update

    if missing_resolve_function is not None:
        model_dict = model.state_dict()
        for key in model_dict.keys():
            if key not in state_dict:
                new_state = missing_resolve_function(key, model_dict[key], state_dict)
                if new_state is not None:
                    state_dict[key] = new_state
                print(f"missing {key}, {model_dict[key].shape}")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if log_missing_keys:
        print('missing_keys', missing_keys)
    print('unexpected_keys', unexpected_keys)

def save_checkpoint(model, output_dir, requires_grad=True):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    _state_dict: Dict[str, Any] = model.state_dict()
    requires_grad_keys = []
    for key, para in model.named_parameters():
        if para.requires_grad:
            requires_grad_keys.append(key)
    requires_grad_keys = set(requires_grad_keys)
    state_dict = {}
    for key, value in _state_dict.items():
        if not requires_grad or key in requires_grad_keys:
            if key.startswith("module."):
                key = key[len("module."):]
            state_dict[key] = value
    torch.save(state_dict, output_dir)
    logger.info(f"Model weights saved in {output_dir}")
    return state_dict


if __name__ == '__main__':
    pass
    conv_model = torch.nn.Conv2d(100, 100, 3, 3)
    conv_model.requires_grad_(False)
    state_dict = save_checkpoint(conv_model, "/home/zhuyixing/tmp/model.bin")
    breakpoint()
    state_dict = {'weight': torch.randn(5, 5, 3, 3), 'bias': torch.randn(5)}
    load_checkpoint(conv_model, state_dict)
    breakpoint()
    pass
