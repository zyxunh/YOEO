import os
from collections import OrderedDict
from typing import Union, List, Tuple

from lmdb import Environment

from unhcv.common.utils import obj_load, obj_dump, get_logger, find_path, attach_home_root
import os.path as osp
import lmdb
from unhcv.common.types import ListDict, ListDictWithIndex
from .file_backend import Backend

logger = get_logger(__name__)


class LmdbWriter:
    def __init__(self, env: Environment, write_commit_num: int):
        self.env = env
        self.write_commit_num = write_commit_num
        self.init_cache()

    def init_cache(self):
        self.cache = OrderedDict()
        self.overwrite = None
        self.write_accumulation_num = 0

    def put(self, name, data, overwrite=False):
        self.cache[name] = data
        self.write_accumulation_num += 1
        self.overwrite = overwrite

        if self.write_accumulation_num >= self.write_commit_num:
            self.commit()
        return True

    def commit(self):
        if len(self.cache):
            with self.env.begin(write=True) as txn:
                for key, value in self.cache.items():
                    flag = txn.put(key, value, overwrite=self.overwrite)
                    if not flag:
                        logger.warn(f"{key} write fail")
        self.init_cache()

class LmdbBackend(Backend):
    def __init__(self, root, root_ids=None, readers=126, mode="r", decode_methods={}, backend_config={}):
        Backend.__init__(self, root=root, mode=mode, decode_methods=decode_methods)
        self.write_commit_num = 100
        self.backend_dict = {}
        self.backend = None
        if isinstance(root, str):
            self.backend = self.init_env(root, mode=mode, readers=readers, **backend_config)
        else:
            roots = root
            for i, root in enumerate(root):
                self.backend_dict[root_ids[i]] = self.init_env(root, mode=mode, readers=readers, **backend_config)
            self.backend = self.backend_dict[root_ids[0]]

    def init_env(self, root, mode, **kwargs):
        max_readers = kwargs.pop("readers")
        kwargs = {}
        if mode == "r":
            root = find_path(root)
            readonly = True
            write = False
        elif mode == "w":
            readonly = False
            write = True
            os.makedirs(osp.dirname(root), exist_ok=True)
            assert not osp.exists(root), root
            kwargs["map_size"] = int(1e12)
            logger.info(f"write data to {root}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        lmdb_env = lmdb.open(
            root,
            max_readers=max_readers,
            readonly=readonly,
            lock=False,
            readahead=False,
            meminit=False,
            **kwargs)

        self.lmdb_env = lmdb_env
        if mode == "w":
            return LmdbWriter(lmdb_env, write_commit_num=self.write_commit_num)
        elif mode == "r":
            return lmdb_env.begin(write=write)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def backend_read(backend, name):
        if isinstance(name, (List, Tuple)):
            out = [backend.get(var.encode()) for var in name]
        else:
            out = backend.get(name.encode())
        return out

    @staticmethod
    def backend_write(backend: Union[open, Environment], name, data):
        if isinstance(data, bytes):
            pass
        else:
            data = obj_dump(name, data, return_buffer=True)
        flag = backend.put(name.encode(), data, overwrite=False)
        if not flag:
            logger.warn(f"{name} write fail")

    def read(self, name):
        if isinstance(name, str):
            root_id = None
            buffer = self.backend_read(self.backend, name)
        else:
            root_id, name = name
            buffer = self.backend_read(self.backend_dict[root_id], name)
        data = self.decode(name, buffer, root_id)
        return data

    def read_buffer(self, name):
        if isinstance(name, str):
            root_id = None
            buffer = self.backend_read(self.backend, name)
        else:
            root_id, name = name
            buffer = self.backend_read(self.backend_dict[root_id], name)
        return buffer

    def read_many(self, names, return_name=False):
        assert not return_name
        name_bucket = ListDictWithIndex()
        for index_name, name in enumerate(names):
            if isinstance(name, str):
                root_id = None
            else:
                root_id, name = name
            name_bucket.append(root_id, name, index_name)
        input_names = names
        names = []
        datas = [[]] * len(input_names)
        for root_id in name_bucket.value_dict.keys():
            _names = name_bucket.value_dict[root_id]
            _indexes_name = name_bucket.index_dict[root_id]
            _datas = self.backend_read(self.backend_dict[root_id] if root_id else self.backend, _names)
            for _index_name, name, data in zip(_indexes_name, _names, _datas):
                if data is None:
                    raise ValueError(f"unable to find {name}")
                datas[_index_name] = self.decode(name, data, root_id)
                # datas.append(self.decode(name, data, root_id))
            names.extend(_names)
        if return_name:
            return names, datas
        # assert list(input_names) == names
        return datas

    def write(self, name, data):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        if isinstance(name, str):
            root_id = None
            backend = self.backend
        else:
            root_id, name = name
            backend = self.backend_dict[root_id]
        self.backend_write(backend, name, data)
        # self.write_commit()

    def write_commit(self):
        raise NotImplementedError
        if self.write_commit_num is None:
            return
        self.write_accumulation_num += 1
        if self.write_accumulation_num > self.write_commit_num:
            self.backend.commit()
            self.write_accumulation_num = 0

    def write_many(self, names, datas):
        self.backend_write(self.backend, names, datas)

    def save(self, name, data):
        raise NotImplementedError

    def commit(self):
        if self.backend_dict:
            backends = self.backend_dict.values()
        else:
            backends = [self.backend]
        for backend in backends:
            backend.commit()

    def end_write(self):
        self.commit()
        self.lmdb_env.close()
