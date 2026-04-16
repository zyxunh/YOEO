from unhcv.common.utils import obj_load, obj_dump, find_path
import os.path as osp


class Backend:
    def __init__(self, root=None, mode="r", decode_methods={}):
        self.root = root
        self.mode = mode
        self.decode_methods = decode_methods
        self.decode_methods_root_ids = decode_methods.get("root_ids", {})
        self.decode_methods_name_suffix = decode_methods.get("name_suffix", {})
        self.decode_methods_default = decode_methods.get("default", None)
        assert not self.decode_methods_name_suffix

    def decode(self, name, data, root_id):
        decode_method = self.decode_methods_root_ids.get(root_id, None)
        if decode_method is not None:
            return decode_method(name, data, root_id)
        if self.decode_methods_default is not None:
            return self.decode_methods_default(name, data, root_id)
        return obj_load(name, data)

    def read(self, name):
        raise NotImplementedError

    def save(self, name, data):
        raise NotImplementedError

    def read_many(self, names, return_name=False):
        return [self.read(name) for name in names]



class FileBackend(Backend):
    def __init__(self, root=None, mode="r", decode_methods={}):
        super().__init__(mode=mode, decode_methods=decode_methods)
        self.root = find_path(root)

    def read(self, name):
        data = obj_load(osp.join(self.root, name))
        return data

    def save(self, name, data):
        obj_dump(osp.join(self.root, name), data)