from .file_backend import Backend
from .lmdb_backend import LmdbBackend
import webdataset
from dataloader import KVWriter, KVReader


class WebBackend(LmdbBackend, Backend):

    def __init__(self, root=None, root_ids=None, readers=8, mode="r", decode_methods={}):
        super().__init__(root=root, root_ids=root_ids, readers=readers, mode=mode, decode_methods=decode_methods)

    def init_env(self, root, mode, **kwargs):
        readers = kwargs.pop("readers")
        env = webdataset.WebDataset(root)
        return env

    @staticmethod
    def backend_read(backend, name):
        return backend.read_many([name])[0]

    def __iter__(self):
        iterator = iter(self.backend)
        while True:
            try:
                data = next(iterator)
            except:
                raise StopIteration
            data = self.decode(None, data, None)
            yield data

    def save(self, name, data):
        raise NotImplementedError
        self.kv_env.write_many([name], [data])

    @property
    def all_keys(self):
        all_keys = self.backend.list_keys()
        sorted(all_keys)
        return all_keys


if __name__ =="__main__":

    import bson
    from unhcv.common.utils import BufferTool, write_im
    from unhcv.common.image import putText, concat_differ_size
    import os.path as osp
    import numpy as np
    def decode_methods_default(name, data, root_id):
        new_data = {}
        image = BufferTool.pil_decode(data['image'])
        new_data['image'] = image
        new_data['text'] = data['caption'].decode()
        new_data['aesthetic'] = float(data['aesthetic_score'].decode())
        new_data['original_key'] = data['original_key'].decode()
        return new_data
    decode_methods = dict(default=decode_methods_default)
    show_root = "/home/tiger/show/laion-5b"
    data_root = "/home/tiger/dataset/BrushData/00001.tar"
    web_backend = WebBackend(root=data_root, decode_methods=decode_methods)
    # keys = kv_backend.all_keys
    import tqdm
    i = 0
    iterator = iter(web_backend)
    while(1):
        i += 1
        print(i)
        data = next(iterator)
        image = np.array(data['image'])[..., ::-1]
        image_text = np.zeros_like(image)
        image_text = putText(image_text, show_texts=["text:" + data['text'] + f"   aesthetic: {data['aesthetic']}"])
        image = concat_differ_size([image, image_text], axis=1)
        write_im(osp.join(show_root, osp.basename(data_root), data['original_key'] + ".jpg"), image)
    breakpoint()
    pass
