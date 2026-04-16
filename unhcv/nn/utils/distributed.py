from typing import Any
from accelerate.utils import gather_object
from unhcv.common.utils import concat_unified


__all__ = ['all_gather_object']


def all_gather_object(object: Any):
    objects = gather_object(object)
    objects = concat_unified(objects)
    return objects