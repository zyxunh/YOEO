import os.path as osp
from typing import Sequence, Union, List, Tuple

import PIL.Image
from functools import partial

from .file_backend import Backend
from .lmdb_backend import LmdbBackend
from dataloader import KVWriter, KVReader
from unhcv.common.fileio.hdfs import listdir, remove_dir
from unhcv.common.utils import get_logger, obj_dump
import msgpack
import bson
from unhcv.common.utils.file import BufferTool

logger = get_logger(__name__)


class KvBackend(LmdbBackend, Backend):

    def __init__(self, root=None, root_ids=None, readers=8, mode="r", decode_methods={},
                 backend_config=dict(max_shard_size=int(1024 ** 3 * 10)), remove_exists=False):
        super().__init__(root=root, root_ids=root_ids, readers=readers, mode=mode, decode_methods=decode_methods,
                         backend_config=dict(**backend_config, remove_exists=remove_exists))
        self.write_cache_num = 0
        self.flush_num = None
        self.end_flag = False

    def init_env(self, root, mode, remove_exists=False, **kwargs):
        if remove_exists:
            for path in listdir(root + "*"):
                print(f'remove {path}')
                remove_dir(path)
        readers = kwargs.pop("readers")
        max_shard_size = kwargs.pop("max_shard_size")
        if mode == "r":
            kv_env = KVReader(root, readers)
            kv_env.list_keys()
        elif mode == "w":
            kv_env = KVWriter(root, max_shard_size)
            print('write to {}'.format(root))
        else:
            raise NotImplementedError
        return kv_env

    @staticmethod
    def backend_read(backend, name):
        if isinstance(name, str):
            return backend.read_many([name])[0]
        return backend.read_many(name)

    @staticmethod
    def backend_write(backend, name, data):
        if isinstance(name, str):
            name = [name]
            data = [data]
        for i, data_i in enumerate(data):
            if not isinstance(data_i, bytes):
                data[i] = obj_dump(name[i], data_i, return_buffer=True)
        return backend.write_many(name, data)

    def write(self, name, data):
        super().write(name, data)
        self.write_cache_num += 1
        if self.flush_num is not None and self.write_cache_num > self.flush_num:
            self.backend.flush()
            self.write_cache_num = 0
            logger.info("flush")

    def __len__(self):
        return len(self.all_keys)

    def __iter__(self):
        for key in self.all_keys:
            yield self.read(key)

    @property
    def all_keys(self):
        all_keys = self.backend.list_keys()
        sorted(all_keys)
        return all_keys

    def end_write(self):
        if self.end_flag:
            return
        self.end_flag = True
        self.backend.flush()
        self.write_cache_num = 0
        logger.info("flush")
        logger.info("end write")

def obj_dump_to_kv(file, obj):
    suffix = osp.splitext(file)[1]
    kv_backend = KvBackend(file, mode="w", remove_exists=True)
    if suffix == ".msgpack":
        for key, value in obj.items():
            kv_backend.write(key, msgpack.packb(value))
    elif suffix == ".bson":
        kv_backend.write("data", bson.dumps(obj))
    else:
        raise NotImplementedError
    kv_backend.end_write()

def decode_method_kv(suffix, name, data, root_id):
    if suffix == ".msgpack":
        data = msgpack.loads(data)
    elif suffix == ".bson":
        data = bson.loads(data)
    else:
        raise NotImplementedError
    return data
def obj_load_from_kv(file):
    suffix = osp.splitext(file)[1]
    kv_backend = KvBackend(file, mode="r", decode_methods=dict(default=partial(decode_method_kv, suffix)))
    keys = kv_backend.all_keys
    if suffix == ".msgpack":
        obj = dict(zip(*kv_backend.read_many(kv_backend.all_keys)))
    elif suffix == ".bson":
        obj = kv_backend.read("data")
    else:
        raise NotImplementedError
    return obj


if __name__ =="__main__":
    pass