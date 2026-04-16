import os

import numpy as np

from unhcv.common import visual_mask, write_im
from unhcv.common.image import concat_differ_size
from unhcv.common.utils import find_path, attach_home_root
from unhcv.datasets.common_datasets import Dataset


def visual_dataset():
    dataset = Dataset(data_indexes_path=find_path("dataset/defacto/indexes.yml"),
                      data_root=find_path("dataset/defacto"), batch_size=1)
    show_root = attach_home_root("show/defacto")
    for i, data in enumerate(dataset):
        if i == 100:
            break
        image = np.array(data['image'].convert('RGB'))[..., ::-1]
        gt = np.array(data['gt'].convert('RGB'))[..., ::-1]
        inpainting_mask = np.array(data['inpainting_mask'].convert('L'))
        inpainting_mask_show = visual_mask(image, inpainting_mask)[-1]
        show = concat_differ_size([image, inpainting_mask_show, gt], 1)
        write_im(os.path.join(show_root, f"{i}.jpg"), show)
        pass


if __name__ == '__main__':
    visual_dataset()