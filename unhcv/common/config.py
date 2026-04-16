from enum import Enum

import functools
import dataclasses
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Type

from pydantic import BaseModel as _BaseModel

from unhcv.common.types import PathStr
from unhcv.common.utils import find_path
from unhcv.common.utils.mmcv_utils import Config


@dataclass(repr=False)
class CfgNode(Mapping):
    new_allowed: bool = False

    def get(self, key, default: Any = None, delete=False) -> Any:
        if isinstance(key, tuple):
            out = []
            for k in key:
                out.append(self.get(k, default))
            return out
        if True or dataclasses.is_dataclass(self):
            if delete:
                return self.__dict__.pop(key, default)
            else:
                return self.__dict__.get(key, default)
            # if key in self:
            #     value = getattr(self, key)
            #     if delete:
            #         self.delete(key)
            #     return value
            # if default is not None:
            #     return default
            # raise KeyError(f"{key} not in {self.__class__.__qualname__}")

    def pop(self, key: str, default: Any = None) -> Any:
        return self.get(key, default=default, delete=True)

    def delete(self, key):
        delattr(self, key)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key, /):
        return self.get(key)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def set(self, key, value):
        if isinstance(key, tuple):
            for k, v in zip(key, value):
                self.set(k, v)

        if True or dataclasses.is_dataclass(self):
            setattr(self, key, value)

    def contains(self, key):
        if True or dataclasses.is_dataclass(self):
            return hasattr(self, key)

    def __contains__(self, key: object, /) -> bool:
        return self.contains(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def merge(self, other, new_allowed=None):
        """
        Merge the contents of another CfgNode into this one.
        """
        if other is None:
            other = {}
        if new_allowed is None:
            new_allowed = self.new_allowed
        if isinstance(other, type) and issubclass(other, CfgNode):
            other = other()
        if isinstance(other, (Mapping, Config)):
            for k, v in other.items():
                if self.contains(k):
                    node_v = self.get(k)
                else:
                    if not new_allowed:
                        raise KeyError(f"Key {k} not found in CfgNode {self.__class__.__qualname__}")
                    node_v = None
                if isinstance(node_v, CfgNode):
                    node_v.merge(v)
                else:
                    self.set(k, v)
        else:
            raise TypeError(f"Expected dict, got {type(other)}")
        if hasattr(self, '__post_init__'):
            self.__post_init__()
        return self

    def __repr__(self):
        return self.__class__.__qualname__ + '(' + ', '.join([f"{f[0]}={f[1]}" for f in self.__dict__.items()]) + ')'

    def __post_init__(self):
        for key, field in self.__dataclass_fields__.items():
            if isinstance(field.type, type) and issubclass(field.type, Enum):
                setattr(self, key, field.type(getattr(self, key)))
            if isinstance(field.type, type) and issubclass(field.type, PathStr):
                setattr(self, key, find_path(getattr(self, key)))

    @classmethod
    def from_other(cls, other):
        if isinstance(other, cls):
            return other
        return cls().merge(other)


class BaseModel(_BaseModel, CfgNode):
    pass


def config_adapt(init_func=None):
    @functools.wraps(init_func)
    def wrapped(self, *args, **kwargs):
        signature = inspect.signature(init_func)
        cfg_node_pos = tuple(signature.parameters.values())[1]
        if issubclass(cfg_node_pos.annotation, CfgNode):
            return init_func(self, *args, **kwargs)
        signature.parameters.values()
        return init_func(self, *args, **kwargs)

    return wrapped

if __name__ == "__main__":
    z = CfgNode()
    z.y = 1
    class CfgNode2(CfgNode):
        c = 3
    z.z = CfgNode2()
    def fun(**kwargs):
        pass

    fun(**z)
    breakpoint()
    dataclasses.asdict()
    breakpoint()

    class Tmp:

        @config_adapt
        def __init__(self, *, x:int = 1, y:CfgNode):
            pass

    k = Tmp(dict(x=2))

    pass