from itertools import islice
import math
import torch


def chunk(it, size, return_list=True):
    it = iter(it)
    it_chunk = iter(lambda: tuple(islice(it, size)), ())
    if return_list:
        return tuple(it_chunk)
    return it_chunk

def split(it, num, return_list=True, drop_last=False, keep_order=False):
    out = []
    it_num = len(it)
    size = it_num / num
    if keep_order:
        for i in range(num):
            out.append(it[i::num])
        return out

    data_func = tuple
    if isinstance(it, torch.Tensor):
        data_func = lambda x: torch.stack(tuple(x))
    assert it_num >= num, 'it_num:{}, num: {}'.format(it_num, num)
    if drop_last:
        it_num = int(math.floor(size)) * num
    size_floor = int(math.floor(it_num / num))
    it_num_extra = round((size - size_floor) * num)
    assert it_num_extra + num * size_floor == it_num, "it_num, num: {}, {}".format(it_num, num)
    it = iter(it)
    for i in range(num):
        if it_num_extra > 0:
            out.append(data_func(islice(it, size_floor + 1)))
            it_num_extra -= 1
        else:
            out.append(data_func(islice(it, size_floor)))
    return out

if __name__ == '__main__':
    import numpy as np
    for num in range(1, 389):
        k = split(np.arange(388), num, keep_order=True)
        if not len(np.unique(np.concatenate(k))) == 388:
            breakpoint()
        num_per = [len(var) for var in k]
        assert max(num_per) - min(num_per) <= 1
    print(k)
    k = split(np.arange(388), 8, keep_order=True)
    breakpoint()
    # tuple(iter(lambda: tuple(islice(iter(np.arange(10)), 3)), ()))
    out = list(chunk([1,2,3,4,5], 2))
    k = list(split(np.arange(95), num=10))