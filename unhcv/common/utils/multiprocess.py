import os
import math
from multiprocessing import Process, Manager
from .custom_logging import get_logger

logger = get_logger(__name__)

def example(mp_idx=0, mp_num=1, manager_dict=None, **kwargs):
    files = []
    for i, file in enumerate(files):
        if i % mp_num != mp_idx:
            continue


def func(mp_idx=0, mp_num=1, **kwargs):
    import torch
    torch.cuda.set_device(mp_idx)
    per_gpu_mp_num = math.ceil(mp_num / torch.cuda.device_count())
    torch.cuda.set_per_process_memory_fraction(1 / per_gpu_mp_num, device=0)
    for i_multi, (key, value) in enumerate([]):
        if i_multi % mp_num != mp_idx:
            continue
        pass


def base_multiprocess_func(func, num_p, args=[], kwargs={}):
    p_list = []
    arnold_worker_num = int(os.environ.get("ARNOLD_WORKER_NUM", 1))
    arnold_id = int(os.environ.get("ARNOLD_ID", 0))
    arnold_num_p = arnold_worker_num * num_p
    with Manager() as manager:
        manager_dict = manager.dict()
        logger.info(f"process num is {arnold_num_p}")
        if arnold_num_p == 1:
            kwargs['mp_idx'] = arnold_id * num_p
            kwargs['mp_num'] = arnold_num_p
            kwargs['manager_dict'] = manager_dict
            func(*args, **kwargs)
            logger.info(f"join begin, process {kwargs['mp_idx']}")
            return
        for i in range(num_p):
            kwargs['mp_idx'] = arnold_id * num_p + i
            kwargs['mp_num'] = arnold_num_p
            kwargs['manager_dict'] = manager_dict
            p = Process(target=func, args=args, 
                            kwargs=kwargs)
            p_list.append(p)
            logger.info(f"join begin, process {kwargs['mp_idx']}")
        for p in p_list:
            p.start()
        
        for p in p_list:
            p.join()
            logger.info("join end")
    return manager_dict

if __name__ == "__main__":
    base_multiprocess_func(func, 4)