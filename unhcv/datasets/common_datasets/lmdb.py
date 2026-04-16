import lmdb
import numpy as np
import cv2
from .dataread import BaseDataRead


class LmdbDataRead(BaseDataRead):
    def __init__(self, image_root: str, key_file: str, max_readers=8) -> None:
        super().__init__()
        with open(key_file, "r") as f:
            self.keys = f.readlines()
        self.keys = [var.strip() for var in self.keys]

        self.lmdb_env = lmdb.open(
            image_root,
            max_readers=max_readers,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.lmdb_txn = self.lmdb_env.begin(write=False)

    def read_next(self):
        raise NotImplementedError

    def read_with_key(self, key):
        bin = self.lmdb_txn.get(key.encode())
        bin = np.frombuffer(bin, dtype=np.uint8)
        data = cv2.imdecode(bin, cv2.IMREAD_UNCHANGED)
        return data

    def __len__(self):
        return len(self.keys)

    def read_i(self, i):
        return self.read_with_key(self.keys[i])


if __name__ == "__main__":
    from unhcv.common.utils import write_im
    import os.path as osp
    save_root = '/home/tiger/tmp/show'
    lmdb_data_read = LmdbDataRead(
        "/mnt/bn/inpainting-bytenas-lq/data/inpainting/ILSVRC2012/lmdb_img_train",
        "/mnt/bn/inpainting-bytenas-lq/data/inpainting/ILSVRC2012/lmdb_img_train_all_keys.txt",
    )
    for i in range(len(lmdb_data_read)):
        img = lmdb_data_read.read_i(i)
        write_im(osp.join(save_root, f'{i}.jpg'), img)
        print(img.shape)
        
