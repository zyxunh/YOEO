import copy
import os
import numpy as np
from typing import Tuple, List, Optional, Union
import PIL.Image as Image
import torch
from torch.utils.data import Dataset as TDataset, IterableDataset as TIterableDataset
from torch.utils.data._utils.collate import collate_tensor_fn
from torchvision.transforms.functional import pil_to_tensor
from line_profiler import profile

from unhcv.common.utils import find_path, obj_load, get_logger, set_seed
from unhcv.common.array import chunk, split
from unhcv.datasets.transforms.torchvision_transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomResizedWHCrop, CenterCrop
from unhcv.core.utils import get_global_size, get_global_rank

from .file_backend import FileBackend
from .lmdb_backend import LmdbBackend
try:
    from .kv_backend import obj_load_from_kv, obj_dump_to_kv, KvBackend
except (ModuleNotFoundError, ImportError):
    pass
from .size_bucket import SizeBucket

logger = get_logger(__name__)


class DatasetShard:
    generator = None
    seed = 1234
    epoch = 0
    actual_id = 0
    actual_size = 1

    def set_seed(self):
        self.generator = torch.Generator().manual_seed(self.seed + self.epoch)
        # logger.info("actual_id is {}, set seed to {}".format(self.actual_id, self.seed + self.epoch))

    def __init__(self):
        self.set_epoch(0)

    def new_epoch(self):
        self.epoch += 1
        logger.info("set epoch to {}".format(self.epoch))
        self.set_seed()

    def set_epoch(self, epoch):
        self.epoch = epoch
        logger.info("set epoch to {}".format(epoch))
        self.set_seed()

    def set_id(self, actual_id, actual_size):
        self.actual_id = actual_id
        self.actual_size = actual_size
        logger.info("actual_id is {}, actual_size is {}".format(actual_id, actual_size))


class Dataset(TIterableDataset):
    backend_type = "File"
    _batched_record_index0_flag = False

    def __init__(self, data_indexes_path: Optional[str] = None, data_root: Optional[str] = None, root_ids=None,
                 collect_keys=None, shuffle=False, debug=False, batch_size=None, backend_config={},
                 parallel_read_num=1, data_indexes_filter=None, name_pair=None, inference=False):
        super().__init__()
        if data_root is None:
            assert data_indexes_path.endswith("_catalog.bson")
            data_root = data_indexes_path[:-len("_catalog.bson")]
        data_indexes_path = find_path(data_indexes_path)
        self.data_indexes_path = data_indexes_path
        if data_indexes_path.startswith("hdfs"):
            self.data_indexes = self.data_information = obj_load_from_kv(data_indexes_path)
        else:
            self.data_indexes = self.data_information = obj_load(data_indexes_path)
        if 'data_root' in self.data_information:
            data_root = self.data_information['data_root']
        if isinstance(self.data_information, dict):
            self.backend_type = self.data_information.pop("backend_type", self.backend_type)
            self.data_indexes = self.data_information.pop("indexes")
        if root_ids is None and isinstance(self.data_information, dict):
            root_ids = self.data_information.get("root_ids", None)
        if data_indexes_filter is not None:
            self.data_indexes = data_indexes_filter(self.data_indexes)
        if self.backend_type == "File":
            self.data_backend = FileBackend(root=data_root, **backend_config)
        elif self.backend_type == "Lmdb":
            self.data_backend = LmdbBackend(root=data_root, root_ids=root_ids, **backend_config)
        elif self.backend_type == "Kv":
            self.data_backend = KvBackend(root=data_root, root_ids=root_ids, **backend_config)
        else:
            raise NotImplementedError

        self.data_root = data_root
        self.collect_keys = collect_keys
        self.shuffle = shuffle
        self.debug = debug
        self.batch_size = batch_size
        self.drop_last = True
        self.parallel_read_num = parallel_read_num
        self.name_pair = name_pair
        self.return_data_index = False
        self.inference = inference

        data_length = len(self)
        if batch_size is not None and self.drop_last:
            extra_num = data_length % self.batch_size
            if extra_num:
                logger.info("{} will drop {}".format(os.path.basename(data_indexes_path), extra_num))
        logger.info("{} length is {}".format(os.path.basename(data_indexes_path), data_length))

    def read_with_data_indexes(self, data_indexes, return_data_index=False):
        num_data_indexes = len(data_indexes)
        # data_index = self.data_indexes[index]

        index_for_recovery_list = []
        key_list = []
        value_list = []
        data = [{} for _ in range(num_data_indexes)]
        for i_data_index, data_index in enumerate(data_indexes):
            for key, value in data_index.items():
                if (isinstance(value, str) and 'text' not in key) or \
                        (isinstance(value, (Tuple, List)) and isinstance(value[0], str)):
                    key_list.append(key)
                    value_list.append(value)
                    index_for_recovery_list.append(i_data_index)
                else:
                    data[i_data_index][key] = value
        value_decoded_list = self.data_backend.read_many(value_list)
        for index_for_recovery, key, value_decoded in zip(index_for_recovery_list, key_list, value_decoded_list):
            if isinstance(value_decoded, dict):
                data[index_for_recovery].update(value_decoded)
            else:
                data[index_for_recovery][key] = value_decoded
        if self.name_pair is not None:
            for data_i in data:
                for name1, name2 in self.name_pair.items():
                    if name2 in data_i:
                        data_i[name1] = data_i[name2]
                        del data_i[name2]
        if return_data_index:
            return data, data_index
        return data

    @profile
    def read_with_index(self, index, return_data_index=False):
        if not isinstance(index, tuple):
            index = (index, )
            index_int_flag = True
        else:
            index_int_flag = False
        data_indexes = [self.data_indexes[var] for var in index]
        data = self.read_with_data_indexes(data_indexes, return_data_index=return_data_index)
        for i_data in range(len(data)):
            data[i_data]['index'] = data_indexes[i_data]
            data[i_data]['index_i'] = index[i_data]
        if return_data_index:
            data, data_index = data
        if self.return_data_index:
            data[0]["data_indexes"] = data_indexes[0]
        if index_int_flag:
            data = data[0]
        if return_data_index:
            return data, data_index
        return data

    def postprocess(self, data):
        if self.debug:
            return data
        if self.collect_keys is not None:
            new_data = {}
            for key in self.collect_keys:
                new_data[key] = data[key]
            data = new_data
        return data

    def __len__(self):
        return len(self.data_indexes)

    def __getitem__(self, index):
        return self.read_with_index(index)

    def batched_record_index0_start(self):
        self._batched_record_index0_flag = True

    def batched_record_index0_end(self):
        self._batched_record_index0_flag = False

    @property
    def batched_record_index0_flag(self):
        return self._batched_record_index0_flag or self.batch_size == 1

    @profile
    def __iter__(self):
        data_length = len(self)
        if self.shuffle:
            indexes = np.random.permutation(data_length)
        else:
            indexes = np.arange(data_length)

        if self.batch_size is not None:
            # assert self.batch_size < data_length
            if self.batch_size > data_length:
                print(
                    "data_root {}'s length is {} batch_size is {}".format(self.data_root, data_length, self.batch_size))
                return
            if self.drop_last:
                extra_num = data_length % self.batch_size
                if extra_num:
                    random_indexes = np.random.permutation(data_length)
                    indexes = np.delete(indexes, random_indexes[:extra_num])

            assert len(indexes) % self.batch_size == 0
            indexes_chunked = chunk(indexes, self.batch_size)
        else:
            raise ValueError("Batch size must be number")
        for indexes in indexes_chunked:
            if self.batch_size is not None:
                self.batched_record_index0_start()
            indexes = chunk(indexes, self.parallel_read_num)
            for i_index, index in enumerate(indexes):
                if self.debug:
                    datas = self.read_with_index(index)
                else:
                    datas = self.read_with_index(index)
                for data in datas:
                    data = self.postprocess(data)
                    yield data

        # for index in indexes:
        #     data = self.read_with_index(index)
        #     yield data


class DatasetWithPreprocess(Dataset):
    def __init__(self, *, data_indexes_path: Optional[str] = None, data_root: Optional[str] = None,
                 transforms_kwargs=dict(), image_keys=("image", "mask"), image_modes=None, collect_keys=None,
                 shuffle=False, debug=False, batch_size=None, backend_config={}, parallel_read_num=None,
                 data_indexes_filter=None, name_pair=None, inference=False):
        if isinstance(image_keys, str):
            image_keys = (image_keys,)
        self.image_keys = image_keys
        self.image_modes = image_modes
        self.transform_method = "default"
        # self.inference = inference
        super().__init__(data_indexes_path=data_indexes_path, data_root=data_root, collect_keys=collect_keys,
                         shuffle=shuffle, debug=debug, batch_size=batch_size, backend_config=backend_config,
                         parallel_read_num=parallel_read_num, data_indexes_filter=data_indexes_filter, name_pair=name_pair, inference=inference)
        self.init_transforms(transforms_kwargs)

    def build_bucket_shape_transforms(self, image):
        hw = self.size_bucket.match((image.height, image.width))
        # hw_ratio = data['image'].height / data['image'].width
        # hw = area2hw(area=self.transform_max_area, hw_ratio=hw_ratio, max_stride=self.max_stride)
        ratio = image.width / image.height
        self.random_resized_crop = RandomResizedCrop(size=hw, ratio=(ratio, ratio), scale=(1, 1),
                                                     interpolations=self.interpolations, random_on=False)

    @profile
    def preprocess(self, data):
        image_dict = {}
        images: List[Image.Image] = []
        for key in self.image_keys:
            images.append(data[key])
        image_modes = self.image_modes
        if image_modes is not None:
            if isinstance(image_modes, str):
                image_modes = [image_modes] * len(images)
            _images = []
            for image, image_mode in zip(images, image_modes):
                if image.mode != image_mode:
                    if (image.mode == "RGBA" or image.info.get("transparency",
                                                               None) is not None) and image_mode == "RGB":
                        image = image.convert("RGBA")
                        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                        white.paste(image, mask=image.split()[3])
                        image = white
                    else:
                        image = image.convert(image_mode)
                _images.append(image)
            images = _images
        if self.batched_resize and self.batched_record_index0_flag:
            self.batched_record_index0_end()
            self.build_bucket_shape_transforms(data['image'])
        if self.transform_method == "default":
            pass
        elif self.transform_method == "resize_to_bucket_shape":
            self.build_bucket_shape_transforms(data['image'])
        if self.inference:
            image_dict['original_hw'] = (images[0].height, images[0].width)
        if self.transform_method == "center_crop":
            images_value = self.random_resized_crop(list(images))
        else:
            images_value = self.random_resized_crop(list(images), self.interpolations)
        image_dict.update(dict(zip(self.image_keys, images_value)))
        return image_dict

    def init_transforms(self, transforms_kwargs):
        interpolations = transforms_kwargs.get("interpolations", ("bicubic", "nearest-exact"))
        if isinstance(interpolations, str):
            interpolations = (interpolations, )
        self.interpolations = interpolations
        self.batched_resize = transforms_kwargs.get("batched_resize", True)
        self.transform_max_area = transforms_kwargs.get("max_area", 512 * 512)
        self.max_stride = transforms_kwargs.get("max_stride", 1)
        size = transforms_kwargs.get("size", 512)
        scale = transforms_kwargs.get("scale", (0.9, 1))
        self.transform_method = transforms_kwargs.get("transform_method", "default")
        if self.transform_method == "center_crop":
            self.random_resized_crop = CenterCrop(size=size)
        else:
            self.random_resized_crop = RandomResizedWHCrop(size=size, scale=scale, ratio=(1, 1), interpolations=interpolations)
        # self.random_flip = RandomHorizontalFlip()
        if self.batched_resize:
            self.size_bucket = SizeBucket(stride=self.max_stride, **transforms_kwargs.get("size_bucket_config", {}))

    @profile
    def read_with_index(self, index):
        if not isinstance(index, tuple):
            index = (index,)
            index_int_flag = True
        else:
            index_int_flag = False
        data = super().read_with_index(index)
        if not isinstance(data, list):
            data = [data]

        for data_i in data:
            if self.inference:
                pass
                # data_raw = copy.deepcopy(data_i)
                # data_i['data_raw'] = data_raw
            image_dict = self.preprocess(data_i)
            if self.debug:
                for key in image_dict.keys():
                    if key in data_i:
                        data_i[key + "_raw"] = data_i[key]
            data_i.update(image_dict)
        if index_int_flag:
            data = data[0]
        return data

    def __getitem__(self, index):
        data = self.read_with_index(index)
        data = self.postprocess(data)
        return data

    def postprocess(self, data):
        for key, value in data.items():
            if isinstance(value, Image.Image):
                data[key] = pil_to_tensor(value)
        data['image_name'] = data['index']['image']
        return super().postprocess(data)


class ConcatDataset(TIterableDataset, DatasetShard):

    def __init__(self, dataset_clses, common_config=None, custom_configs=None, read_mode="one_by_one", choice_p=None,
                 max_per_read=None, infinite=True):
        DatasetShard.__init__(self)
        self.custom_configs = custom_configs
        self.dataset_clses = dataset_clses
        self.common_config = common_config
        self.read_mode = read_mode
        self.choice_p = choice_p
        self.datasets = []
        self.per_dataset_length = []
        self.max_per_read = max_per_read
        self.infinite = infinite

        if isinstance(dataset_clses, tuple):
            raise NotImplementedError("dataset_clses tuple not yet implemented")
            for dataset in dataset_clses:
                self.datasets.append([dataset, iter(dataset)])
                self.per_dataset_length.append(len(dataset))
        else:
            self.num_dataset = len(custom_configs)
            if read_mode == "random_choice":
                raise NotImplementedError("random_choice not yet implemented")
                for i_dataset in range(len(custom_configs)):
                    dataset_cls = self.dataset_clses[i_dataset] if isinstance(self.dataset_clses,
                                                                              tuple) else self.dataset_clses
                    dataset = dataset_cls(**self.common_config, **self.custom_configs[i_dataset])
                    self.datasets.append([dataset, iter(dataset)])
                    self.per_dataset_length.append(len(dataset))
            elif read_mode == "one_by_one":
                self.per_dataset_length = [var.pop("length") for var in custom_configs]
        logger.info("num_dataset is {}, len_dataset is {}".format(self.num_dataset, len(self)))

    def __len__(self):
        return sum(self.per_dataset_length)

    def __iter__(self):
        if self.read_mode == "one_by_one":
            while True:
                randperm_indexes = torch.randperm(self.num_dataset, generator=self.generator)
                logger.info(
                    f"actual_id is {self.actual_id}, epoch is {self.epoch}, randperm_indexes first 5 {randperm_indexes[:5]}")
                randperm_indexes = split(randperm_indexes, self.actual_size)[self.actual_id]
                # logger.info(f"sharder id: {self.actual_id}, size: {self.actual_size}")
                for i_dataset in randperm_indexes:
                    dataset_cls = self.dataset_clses[i_dataset] if isinstance(self.dataset_clses, tuple) else self.dataset_clses
                    dataset = dataset_cls(**self.common_config, **self.custom_configs[i_dataset])
                    readed_num = 0
                    try:
                        dataset_iter = iter(dataset)
                        while True:
                            try:
                                data = next(dataset_iter)
                            except StopIteration:
                                break
                            yield data
                            readed_num += 1
                            if self.max_per_read is not None and readed_num >= self.max_per_read:
                                break
                    except Exception as ex:
                        logger.error("data_root {} read error, error is {}".format(
                            self.custom_configs[i_dataset]["data_root"], ex))
                        import traceback
                        traceback.print_exc()

                self.new_epoch()
                if not self.infinite:
                    break

        elif self.read_mode == "random_choice":
            raise NotImplementedError
            dataset = np.random.choice(self.datasets, self.choice_p)
            try:
                data = next(dataset[1])
            except StopIteration:
                dataset[1] = iter(dataset[0])
                data = next(dataset[1])
            yield data
        else:
            raise NotImplementedError

    def set_id(self, actual_id, actual_size):
        super().set_id(actual_id, actual_size)


class ShardWrapper(TIterableDataset, DatasetShard):

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        while True:
            if self.random_shuffle:
                randperm_indexes = torch.randperm(len(self), generator=self.generator)
            else:
                randperm_indexes = torch.arange(0, len(self))
            if self.max_num is not None:
                randperm_indexes = randperm_indexes[:self.max_num]
            logger.info(
                f"actual_id is {self.actual_id}, epoch is {self.epoch}, randperm_indexes first 5 {randperm_indexes[:5]}")
            randperm_indexes = split(randperm_indexes, self.actual_size, keep_order=True)[self.actual_id]
            logger.info(f"child randperm_indexes first 5 {randperm_indexes[:5]}")

            for index in randperm_indexes:
                if self.fix_seed_for_all:
                    if not self._debug:
                        assert self.actual_size > 1, "actual_size is {}, data is {}".format(self.actual_size, self.dataset)
                    set_seed(index.item(), set_torch=True)
                yield self.dataset.__getitem__(index)
            if self.infinite:
                self.new_epoch()
            else:
                break

    def __init__(self, dataset, *, random_shuffle=False, drop_last=False, infinite=False, max_num=None, fix_seed_for_all=False):
        DatasetShard.__init__(self)
        self.dataset = dataset
        self.random_shuffle = random_shuffle
        self.drop_last = drop_last
        self.infinite = infinite
        self.max_num = max_num
        self.fix_seed_for_all = fix_seed_for_all
        self._debug = False

    def debug(self):
        self._debug = True


class ShardWrapperTrain2Test(ShardWrapper):

    def __init__(self, dataset, *, random_shuffle=False, drop_last=False, infinite=False, num=None):
        self.num = num
        assert not random_shuffle and not drop_last and not infinite
        super().__init__(dataset, random_shuffle=random_shuffle, drop_last=drop_last, infinite=infinite)

    def __len__(self):
        if self.num is not None:
            return self.num
        return len(self.dataset)

    def __iter__(self):
        randperm_indexes = torch.randperm(len(self.dataset), generator=self.generator)
        if self.num is not None:
            randperm_indexes = randperm_indexes[:self.num]
        randperm_indexes = torch.sort(randperm_indexes)[0]
        logger.info(f"actual_id is {self.actual_id}, epoch is {self.epoch}, randperm_indexes first 5 {randperm_indexes[:5]}")
        randperm_indexes = split(randperm_indexes, self.actual_size, keep_order=True)[self.actual_id]
        for index in randperm_indexes:
            yield self.dataset.__getitem__(index)


def sharder_worker_init_fn(worker_id, seed=None):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    global_size = get_global_size()
    global_rank = get_global_rank()
    num_workers = worker_info.num_workers
    actual_size = global_size * num_workers
    actual_id = global_rank * num_workers + worker_id
    if isinstance(dataset, DatasetShard):
        dataset.set_id(actual_id=actual_id, actual_size=actual_size)
    if seed is not None:
        set_seed(seed + actual_id)
        logger.info(f"set process_{actual_id:02} seed to {seed + actual_id}")


def custom_collate_tensor_fn(batch, *, collate_fn_map):
    # 检查所有张量是否有相同的形状
    shapes = [item.shape for item in batch]
    if all(s == shapes[0] for s in shapes):
        return collate_tensor_fn(batch, collate_fn_map=collate_fn_map)
    else:
        return batch  # 返回原始列表

def custom_collate_list(batch, *args, **kwargs):
    return batch


if __name__ == "__main__":
    class FixedTensorDataset(Dataset):
        def __init__(self, num_samples=100, tensor_shape=(1,), **kwargs):
            """
            Toy Dataset 生成固定形状的随机 Tensor
            参数:
                num_samples: 数据集样本数 (默认: 100)
                tensor_shape: 生成 Tensor 的形状 (默认: 3x64x64 模拟图像)
                num_classes: 分类类别数 (默认: 10)
            """
            self.num_samples = num_samples
            self.tensor_shape = tensor_shape

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return dict(x=torch.zeros(self.tensor_shape) + idx)

    dataset = ShardWrapper(FixedTensorDataset(), random_shuffle=True, drop_last=False, infinite=True)
    dataloader = DataLoader(dataset=dataset, batch_size=6, num_workers=6, prefetch_factor=1,
                            collate_fn=torch.utils.data.default_collate, pin_memory=True, shuffle=False,
                            drop_last=False, worker_init_fn=sharder_worker_init_fn)
    out = []
    for i, var in enumerate(dataloader):
        print(var)
        if i % 16 == 0:
            breakpoint()
        out.append(var['x'])
    out = torch.cat(out).unique()
    breakpoint()
