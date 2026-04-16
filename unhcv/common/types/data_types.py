import copy
from dataclasses import dataclass, field
from typing import Tuple, Any, overload

from diffusers.utils import BaseOutput


__all__ = ['DataDict', 'ListDict', 'ListDictWithIndex', 'DataClass', 'default_factory', 'add_base_cls', 'ModelOutput', 'PathStr']


class ListDictWithIndex:
    def __init__(self):
        self.value_dict = {}
        self.index_dict = {}

    def append(self, key, value, index):
        data_list = self.value_dict.get(key, [])
        index_list = self.index_dict.get(key, [])
        if len(data_list) == 0:
            self.value_dict[key] = data_list
            self.index_dict[key] = index_list
        data_list.append(value)
        index_list.append(index)

class ListDict(dict):
    def append(self, key, value):
        data_list = self.get(key, [])
        if not data_list:
            self[key] = data_list
        data_list.append(value)

class ModelOutput(BaseOutput):
    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

class DataDict(dict):
    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    @overload
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value
        return self

    @overload
    def update(self, data_dict):
        self.update(data_dict)
        return self

    def update(self, _data_dict=None, **kwargs):
        if _data_dict is not None:
            super().update(_data_dict)
        else:
            for key, value in kwargs.items():
                self[key] = value
        return self

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        # if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
        super().__setitem__(name, value)
        # super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        # super().__setattr__(key, value)

    def __getattr__(self, item):
        return super().__getitem__(item)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
    
    def __iter__(self):
        return iter(self.values())

    def get_keys(self, *keys):
        for key in keys:
            yield self.get(key, None)


@dataclass
class DataClass:
    def update(self, data_class=None, /,  **kwargs):
        if data_class is None:
            data_class = kwargs
        if not isinstance(data_class, dict):
            data_class = data_class.__dict__
        for key, value in data_class.items():
            setattr(self, key, value)
        return self

    def keys(self):
        return self.__dict__.keys()


def default_factory(data):
    if isinstance(data, type):
        data = data()
    return field(default_factory=lambda: copy.deepcopy(data))


def add_base_cls(*new_bases, name=None):
    if name is None:
        name = new_bases[-1].__name__
    return type(name, new_bases, {})


class PathStr(str): pass


if __name__ == '__main__':
    x = DataDict(x=1)
    x.update(y=2, z=3)
    x.update(DataDict(xx=2))
    pass