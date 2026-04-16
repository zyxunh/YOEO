from dataloader import KVWriter, KVReader, merge
import pickle
import json
import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
import copy
import bson

if not '5.0' in pickle.compatible_formats:
    import pickle5 as pickle

p2k = lambda path, prefix: path.replace(prefix, "").lstrip('/')
encode = lambda k: str(k).encode()

class CONVERTER(object):
    @staticmethod
    def PMASK_TO_NDBYTES(mask_path: str, verbose: bool=False):
        ndarray_mask = np.asarray(Image.open(mask_path).convert('P'))
        if verbose:
            print("mask ids {}".format(np.unique(ndarray_mask)))
        bytes_mask = cv2.imencode('.png', ndarray_mask)[1].tobytes()
        return bytes_mask

    @staticmethod
    def LMASK_TO_NDBYTES(mask_path: str, verbose: bool=False):
        ndarray_mask = np.asarray(Image.open(mask_path).convert('L'))
        if verbose:
            print("mask ids {}".format(np.unique(ndarray_mask)))
        bytes_mask = cv2.imencode('.png', ndarray_mask)[1].tobytes()
        return bytes_mask

class KVDBDataset(object):
    """默认是Reader，如果需要可以设置为Writer。"""
    DECODER_IDENTITY = staticmethod(lambda v: v)
    DECODER_PICKLE = staticmethod(lambda v: pickle.loads(v))
    DECODER_JSON = staticmethod(lambda v: json.loads(v))
    DECODER_STRING = staticmethod(lambda k: str(k, encoding="utf-8"))
    ENCODER_NDARRAY = staticmethod(lambda v, t: cv2.imencode('.'+t, v)[1])
    DECODER_NDARRAY = staticmethod(lambda v: cv2.imdecode(
                        np.frombuffer(v, dtype='uint8'), cv2.IMREAD_UNCHANGED)
    )

    def __init__(self, kv_path,
                    readonly=True,
                    num_parallel_reader=1,
                    num_shard_writer=1,
                    return_key=False,
                    insert_pool_size=1,
                    ex_attribute_keys=[],
                    in_attribute_keys=[],
                    simultaneously_read_num=1
        ):           
        self.__readonly = readonly
        self.__flushed = False

        if readonly:
            self.__handler = KVReader(kv_path, num_parallel_reader)
            self.exist_keys = sorted(self.__handler.list_keys())
            self.ex_attribute_keys = set(ex_attribute_keys)
            self.in_attribute_keys = set(in_attribute_keys)
            self.attribute_keys = set(in_attribute_keys + ex_attribute_keys)
            self.simultaneously_read_num = simultaneously_read_num
            self.init_reader_keys()
        else:
            # self.__handler = KVReader(kv_path, num_parallel_reader)
            # self.exist_keys = copy.deepcopy(sorted(self.__handler.list_keys()))
            # del self.__handler
            self.__handler = KVWriter(kv_path, num_shard_writer)
        self.return_key = return_key
        self.insert_pool_size = insert_pool_size
        self.insert_pool = []

    def init_reader_keys(self):
        self.exist_keys_set = set(self.exist_keys)
        if len(self.attribute_keys):
            keys = []
            for key in self.exist_keys:
                flag = False
                for attribute in self.attribute_keys:
                    if key.startswith(attribute):
                        flag = True
                        break
                if flag == False:
                    keys.append(key)
        else:
            keys = self.exist_keys
        self.exist_keys = keys

    def insert_ndarray(self, k, v, enc_type="png"):
        assert isinstance(v, np.ndarray), f"[ERROR] value {type(v)} must be np.array..."
        image_byte = self.ENCODER_NDARRAY(v, enc_type)
        self.insert(k, image_byte)
    
    def insert_rbfile(self, k, path):
        assert os.path.exists(path), f"[ERROR] path {path} must exists..."
        with open(path, "rb") as f:
            file_byte = f.read()
        self.insert(k, file_byte)

    def insert(self, k: str, v: bytes):
        # if k in self.exist_keys:
        #     return 
        assert isinstance(k, str), f"{k} must be a string..."
        if k.startswith('b\''):
            raise ValueError("[ERROR] key {} error...".format(k))
        self.__handler.write_many([k], [v])

    def insert_many(self, ks, vs):
        assert isinstance(ks, (list, tuple)) and isinstance(vs, (list, tuple)), f"[ERROR] ks/vs must be list..."
        assert isinstance(ks[0], str) and isinstance(vs[0], bytes), f"[ERROR] ks[0]/vs[0] must be str/bytes..."
        self.__handler.write_many(ks, vs)

    def get_attribute_key(self, k, attribute_key):
        if "_".join([attribute_key, k]) in self.exist_keys_set:
            k = "_".join([attribute_key, k])
        else:
            k = "_".join([attribute_key.split("_")[0], k])
        return k

    def get(self, k, attribute_key=None):
        assert isinstance(k, str), f"{k} must be a string..."
        if attribute_key is not None:
            k = self.get_attribute_key(k, attribute_key)
        v = self.__handler.read_many([k])[0]
        return v

    def get_many(self, ks):
        assert isinstance(ks, list) and isinstance(ks[0], str), f"[ERROR] keys {ks} illegal..."
        vs = self.__handler.read_many(ks)
        return vs

    def commit(self):
        assert self.__readonly == False, f"[ERROR] commit can only be called when readonly..."
        self.__handler.flush()
        self.__flushed = True

    def get_ndarray(self, k):
        v = self.get(k)
        v = self.DECODER_NDARRAY(v)
        return v
    
    def get_ndarrays(self, ks):
        vs = self.get_many(ks)
        vs = [self.DECODER_NDARRAY(_) for _ in vs]
        return vs

    def get_pickle(self, k):
        v = self.get(k)
        v = self.DECODER_PICKLE(v)
        return v

    def get_json(self, k):
        v = self.get(k)
        v = self.DECODER_JSON(v)
        return v
    
    def get_obj(self, k):
        ext = osp.splitext(k)[1]
        if ext == '.json':
            return self.get_json(k)
        elif ext == '.pkl':
            return self.get_pickle(k)
        else:
            raise NotImplementedError

    def query(self, k):
        assert isinstance(k, str), f"{k} must be a string..."
        if k in self.exist_keys:
            return True
        else:
            return False

    def get_bson(self, k, attribute_key=None):
        v = bson.loads(self.get(k))
        for attribute_key in self.in_attribute_keys:
            v.update(bson.loads(self.get(k, attribute_key)))
        return v

    def __getitem__(self, index):
        k = self.exist_keys[index]
        if self.return_key:
            return self.get(k), k
        return self.get(k)

    def __len__(self):
        return len(self.exist_keys)

    # def __del__(self):
    #     if not (self.__readonly or self.__flushed):
    #         self.commit()

if __name__ == "__main__":
    # dataset = KVDBDataset(kv_path="/home/tiger/tmp/debug/0000", readonly=False)
    # dataset.insert("image1", "4".encode())
    # dataset.commit()
    # breakpoint()

    dataset = KVDBDataset(kv_path="/home/tiger/tmp/debug/0000", readonly=True)
    dataset.get("image")
    breakpoint()

    dataset = KVDBDataset(kv_path="hdfs://haruna/home/byte_labcv_gan/bang/data/laion_small_pic_collect/laion_2b_en_buckets/002559-10000-576-448_00085910", readonly=True)
    breakpoint()
    dataset = KVDBDataset(kv_path="/home/tiger/SAM_debug", readonly=True)
    # dataset.insert("a", "a_content".encode())
    # dataset.commit()
    # dataset.insert("b", "b_content".encode())
    # dataset.commit()
    # dataset.insert("c", "b_content".encode())
    # dataset.commit()
    
    # dataset.insert("b", "b_content".encode())