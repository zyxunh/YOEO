from dataclasses import dataclass
from typing import Dict, Optional, List

from PIL import Image
from scipy.optimize import root_scalar

from unhcv.common import get_related_path, remove_dir
from unhcv.common.utils import find_path, obj_dump, attach_home_root
from unhcv.datasets.common_datasets import Dataset, LmdbBackend
# from unhcv.datasets.common_datasets.tools.attach_data import ReadData


class PackageData:
    def __init__(self, root, root_ids=None, index_path=None, backend_type="Lmdb", root_ids_overload=None):
        self.backend_type = backend_type
        if backend_type == "Lmdb":
            self.backend = LmdbBackend(root=root, root_ids=root_ids, mode="w")
        else:
            raise NotImplementedError(f"Dataset {backend_type} not implemented")
        self.indexes = []
        if index_path is None:
            if not isinstance(root, str):
                root = root[0]
            index_path = root + "_catalog.bson"
        self.index_path = index_path
        self.root_ids_overload = root_ids_overload
        self.root_ids = root_ids

    def write_indexes(self):
        indexes_dict = dict(backend_type=self.backend_type, indexes=self.indexes)
        if self.root_ids is not None:
            if self.root_ids_overload is None:
                self.root_ids = ['default', *self.root_ids]
            else:
                self.root_ids = self.root_ids_overload + self.root_ids
            indexes_dict['root_ids'] = self.root_ids
        obj_dump(self.index_path, indexes_dict)

    def write_data_dict(self, data_dict, index):
        for key in data_dict.keys():
            name = index[key]
            try:
                self.backend.write(name, data_dict[key])
            except:
                breakpoint()
        self.indexes.append(index)

    def end_write(self):
        self.backend.end_write()
        self.write_indexes()


if __name__ == "__main__":
    # root = attach_home_root("dataset/tmp/package_data_debug")
    # remove_dir(root)
    # package_data = PackageData(root)
    # for i in range(1000):
    #     index = dict(num=f"{i}")
    #     data_dict = dict(num=str(i).encode())
    #     package_data.write_data_dict(data_dict, index)
    # package_data.end_write()

    # root = attach_home_root("dataset/tmp/package_data_debug")
    read_data = ReadData(data_indexes_path='/home/yixing/dataset/tmp/package_data_debug_catalog.bson')
    read_data.read_data_i(900)
    pass
