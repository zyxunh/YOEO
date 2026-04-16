from dataclasses import dataclass
from typing import Dict, Optional, List

from PIL import Image
from scipy.optimize import root_scalar

from unhcv.common import get_related_path
from unhcv.common.utils import find_path, obj_dump, attach_home_root
from unhcv.datasets.common_datasets import Dataset, LmdbBackend
from .package_data import PackageData

class ReadData:
    def __init__(self, data_indexes_path: Optional[str]=None, data_root: Optional[str]=None, root_ids=None):
        self.dataset = Dataset(data_indexes_path=data_indexes_path, data_root=data_root, root_ids=root_ids)
        self.data_indexes = self.dataset.data_indexes
        self.backend_type = self.dataset.backend_type

    def read_data_i(self, i) -> (Dict, Dict):
        data_index = self.data_indexes[i]
        data_dict = self.dataset.read_with_data_indexes([data_index])[0]
        return data_dict, data_index

    def __len__(self):
        return len(self.dataset)


@dataclass
class AttachDataConfig:
    data_indexes_path: Optional[str] = None
    data_root: Optional[str] = None
    save_keys: List = None
    remove_keys: List = None
    save_root: str = None
    save_root_ids: str = None


class AttachData:
    def __init__(self, config: AttachDataConfig):
        self.save_keys = set(config.save_keys)
        self.remove_keys = config.remove_keys
        self.save_root = config.save_root
        self.save_root_ids = config.save_root_ids

        self.read_data = ReadData(data_indexes_path=config.data_indexes_path, data_root=config.data_root)
        if self.save_root_ids is not None:
            pass

        if config.save_root is not None:
            self.package_data = PackageData(root=config.save_root, root_ids=self.save_root_ids, backend_type=self.read_data.backend_type)
        self.new_data_indexes = []

    def __len__(self):
        return len(self.read_data)

    def save_data(self, data_dict, data_index):
        for key in self.remove_keys:
            del data_index[key]
        if self.save_keys is not None:
            new_data_dict = {}
            for key in self.save_keys:
                if key == "image":
                    data = self.read_data.dataset.data_backend.read_buffer(data_index['image'])
                else:
                    data = data_dict[key]
                new_data_dict[key] = data
                if isinstance(data, Image.Image):
                    assert data.mode != "RGB"
                # if key not in data_index:
                if isinstance(data, Image.Image):
                    if data.mode == 'L':
                        data_index[key] = get_related_path(data_index['image'], 'image', key, '.png')
                    else:
                        raise NotImplementedError
                else:
                    breakpoint()
                    raise NotImplementedError
                if self.save_root_ids is not None:
                    data_index[key] = [self.save_root_ids[0], data_index[key]]
            self.package_data.write_data_dict(new_data_dict, data_index)

    def attach_data_dict(self, data_dict, data_index):
        data_dict['mask_1'] = data_dict['mask']
        return data_dict

    def end_write(self):
        self.package_data.end_write()

    def __iter__(self):
        for i in range(len(self)): # len(self)
            # if i != 29:
            #     continue
            data_dict, data_index = self.read_data.read_data_i(i)
            data_dict = self.attach_data_dict(data_dict, data_index)
            self.save_data(data_dict, data_index)
            yield data_dict
        # self.end_write()


if __name__ == '__main__':
    # read_data = ReadData(data_indexes_path=find_path("dataset/open-images-dataset/train/lmdb_1023/086_wh1.333_num10000_catalog.bson"),
    #                      data_root=None)
    # breakpoint()
    # attach_data = AttachData(
    #     AttachDataConfig(data_indexes_path=find_path("dataset/open-images-dataset/train/lmdb/086_wh1.333_num10000_index.bson"),
    #     data_root=find_path("dataset/open-images-dataset/train/lmdb/086_wh1.333_num10000"),
    #     save_root=[attach_home_root("dataset/open-images-dataset/train/lmdb_1023/086_wh1.333_num10000")],
    #     save_root_ids=["new_mask_1"],
    #     save_keys=('mask_1',), remove_keys=("mask_score",)))
    # attach_data_iter = iter(attach_data)
    # for i in range(len(attach_data)):
    #     next(attach_data_iter)

    read_data = ReadData(data_indexes_path=attach_home_root(
        "dataset/open-images-dataset/train/lmdb_1025/000_wh0.25_num10000_catalog.bson"),
             data_root=[find_path("dataset/open-images-dataset/train/lmdb/000_wh0.25_num10000"),
                        find_path("dataset/open-images-dataset/train/lmdb_1025/000_wh0.25_num10000")],
             root_ids=None) #["default", "new_mask_1"]

    for i in range(len(read_data)):
        data = read_data.read_data_i(i)
        breakpoint()
        pass
