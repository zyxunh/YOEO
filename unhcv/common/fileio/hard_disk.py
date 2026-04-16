from unhcv.common.utils import walk_all_files_with_suffix
import os.path as osp


from .kvdb import KVDBDataset

class HarDiskDataset(KVDBDataset):
    def __init__(self, path,
                    readonly=True,
                    **kwargs
        ):          
        self.path = path 
        self.exist_keys = walk_all_files_with_suffix(path, None)
        self.exist_keys = [var[len(path) + 1: ] for var in self.exist_keys]
        assert readonly

    def get(self, k):
        assert isinstance(k, str), f"{k} must be a string..."
        with open(osp.join(self.path, k), "rb") as f:
            file_byte = f.read()
        return file_byte

    def get_many(self, ks):
        assert isinstance(ks, list) and isinstance(ks[0], str), f"[ERROR] keys {ks} illegal..."
        vs = [self.get(k) for k in ks]
        return vs

    def query(self, k):
        assert isinstance(k, str), f"{k} must be a string..."
        if k in self.exist_keys:
            return True
        else:
            return False

    def __del__(self):
        pass

if __name__ == "__main__":
    dataset = HarDiskDataset('/home/tiger/workspace/datasets/sam_640/sa_000993_debug', readonly=True)
    # dataset.insert("a", "a_content".encode())
    # dataset.commit()
    # dataset.insert("b", "b_content".encode())
    # dataset.commit()
    # dataset.insert("c", "b_content".encode())
    # dataset.commit()
    
    # dataset.insert("b", "b_content".encode())