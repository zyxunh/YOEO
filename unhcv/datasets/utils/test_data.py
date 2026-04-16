import torch
import math
import accelerate
import numpy as np
import glob
import os
import os.path as osp
from torchvision import transforms
from torch.utils.data import IterableDataset
from unhcv.common.utils import obj_load
import random

class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:  # single-process data loading, return the full iterator
        #     iter_start = self.start
        #     iter_end = self.end
        # else:  # in a worker process
        #     # split workload
        #     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     iter_start = self.start + worker_id * per_worker
        #     iter_end = min(iter_start + per_worker, self.end)
        for var in range(self.start, self.end):
            print('random random.randint', random.randint(0, 100))
            print('random np.random.randint', np.random.randint(0, 100))
            yield var

class Train2TestIterableDataset(IterableDataset):
    def __init__(self, train_dataset, seed=1234, num=100):
        super().__init__()
        self.train_dataset = train_dataset
        self.seed = seed
        self.num = num

    def __iter__(self):
        # print('set all seed to {}'.format(self.seed))
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        num_readed = 0
        for data in self.train_dataset:
            yield data
            num_readed += 1
            if num_readed >= self.num:
                break

def get_train2test_iterabledataloader(train_dataset, num=50):
    test_dataset = Train2TestIterableDataset(train_dataset, num=num)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=1, worker_init_fn=ConstantSeedWorkerInit().worker_init_fn)
    return test_dataloader

class ConstantSeedWorkerInit:
    def __init__(self, seed=1234) -> None:
        self.seed = seed
    def worker_init_fn(self, worker_id):
        print('set all seed to {}'.format(self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)
        # torch.manual_seed(self.seed)

class LocalTestIterableDataset(IterableDataset):

    DEFAULT_MEAN = [0.5, 0.5, 0.5]
    DEFAULT_STD = [0.5, 0.5, 0.5]
    def __init__(self,
                 root,
                 image_files=None,
                 json_files=None,
                 image_preprocess_param_dict={},
                 max_num=None, train_debug=False):
        super().__init__()
        self.root = root
        if image_files is None:
            image_files = os.listdir(root)
        if json_files is not None:
            image_files = [var for var in image_files if var not in json_files]
        else:
            json_files = []
        self.image_files = image_files
        self.json_files = json_files
        self.key_list = [osp.splitext(var)[0] for var in os.listdir(osp.join(root, image_files[0]))]
        if max_num is not None:
            self.key_list = self.key_list[:max_num]
        default_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(self.DEFAULT_MEAN, self.DEFAULT_STD),
            # transforms.Normalize()
        ])
        transform_dict = {}
        for image_file in image_files:
            if image_file in image_preprocess_param_dict:
                image_preprocess_param = image_preprocess_param_dict[image_file]
                transforms_lt = [
                    transforms.ToTensor(),
                    transforms.Normalize(image_preprocess_param['mean'],
                                         image_preprocess_param['std']),
                ]
                size_param = image_preprocess_param.get("Resize", None)
                if size_param is not None:
                    transforms_lt.append(transforms.Resize(**size_param))
                transform_dict[image_file] = transforms.Compose(transforms_lt)
            else:
                transform_dict[image_file] = default_transform
        self.transform_dict = transform_dict
        self.train_debug = train_debug

    def iter_list(self, key_list):
        for key in key_list:
            batch = dict()
            for file_name in self.image_files:
                batch[file_name] = self.transform_dict[file_name](
                    obj_load(osp.join(self.root, file_name, key) + '.jpg'))
            for file_name in self.json_files:
                batch.update(obj_load(osp.join(self.root, file_name, key) + '.json'))
            yield batch

    def __iter__(self):
        if self.train_debug:
            while True:
                key_list = self.key_list.copy()
                random.shuffle(key_list)
                iterator = self.iter_list(key_list)
                for batch in iterator:
                    yield batch
        else:
            iterator = self.iter_list(self.key_list)
            for batch in iterator:
                yield batch


if __name__ == '__main__':
    if 0:
        data = LocalTestIterableDataset(
            '/home/tiger/test_data/laion_ip_adapter_clip_size_512_num300',
            json_files=['inform'],
            image_preprocess_param_dict=dict(clip_image=dict(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
                Resize=dict(size=448)),
            randommask=dict(mean=[0], std=[1]),
            refer_mask=dict(mean=[0], std=[1])), train_debug=True, max_num=None)
        for batch in data:
            breakpoint()
            pass

    if 1:
        ds = MyIterableDataset(1, 20)
        ds_test = Train2TestIterableDataset(ds, num=3)
        # print(list(torch.utils.data.DataLoader(ds_test, num_workers=0)))
        print(list(torch.utils.data.DataLoader(ds, num_workers=1, worker_init_fn=ConstantSeedWorkerInit().worker_init_fn)))
        print('random random.randint', random.randint(0, 100))
        print('random np.random.randint', np.random.randint(0, 100))
        breakpoint()
        # print(list(torch.utils.data.DataLoader(ds, num_workers=1, worker_init_fn=ConstantSeedWorkerInit().worker_init_fn)))
