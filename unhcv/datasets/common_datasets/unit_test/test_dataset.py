from unhcv.common.utils import obj_dump, get_base_name, attach_home_root
from unhcv.datasets.common_datasets.dataset import Dataset
import os.path as osp
import os


# data_root = "hdfs://haruna/dp/mloops/datasets/unh/dataset/openimage/entity_seg_170w_caption/171-10000-0.375-1"
# data_indexes_root = "hdfs://haruna/dp/mloops/datasets/unh/dataset/openimage/entity_seg_170w_caption/171-10000-0.375-1_catalog.bson"
# show_root = "/home/tiger/show/entity_seg"

data_root = attach_home_root("dataset/open-images-dataset/train/lmdb/008_wh0.667_num10000")
data_indexes_root = data_root + "_index.bson"
show_root = attach_home_root("show")

dataset = Dataset(data_indexes_path=data_indexes_root, data_root=data_root, batch_size=1, debug=True)

data_iter = iter(dataset)
for i_data, data in enumerate(data_iter):
    obj_dump(osp.join(show_root, "image", "{:05}.jpg".format(i_data)), data['image'])
    obj_dump(osp.join(show_root, "mask", "{:05}.png".format(i_data)), data['mask'])
    pass

pass