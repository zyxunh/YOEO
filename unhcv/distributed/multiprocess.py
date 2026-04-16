import torch
import torch.distributed
import os
import functools
from .distributed_inform import get_global_rank, get_global_size

def torch_multiprocess_wrap(func):
    @functools.wraps(func)
    def wrap_func(rank, *args, nprocs=1, **kwargs):
        torch.distributed.init_process_group(
            backend="nccl", init_method="tcp://localhost:12345", world_size=nprocs, rank=rank
        )
        print('global_rank', get_global_rank())
        print('global_size', get_global_size())
        return func(*args, **kwargs)
    return wrap_func

def torch_multiprocess(rank, world_size, *args, **kwargs):
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:12345", world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank)
    print('global_rank', get_global_rank())
    print('global_size', get_global_size())
    x = torch.tensor(0).cuda()
    print(x.device)
    return

class Tmp:
    def func():
        print(1)
        pass
    @classmethod
    def torch_multiprocess(cls, rank, world_size, *args, **kwargs):
        torch.distributed.init_process_group(
            backend="nccl", init_method="tcp://localhost:12345", world_size=world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        print('global_rank', get_global_rank())
        print('global_size', get_global_size())
        x = torch.tensor(0).cuda()
        print(x.device)
        cls.func()
        return

if __name__ == "__main__":
    # @torch_multiprocess_wrap
    def func():
        pass
    class Tmp2(Tmp):
        def func():
            print(2)

    # func_wrap = torch_multiprocess_wrap(func)
    torch.multiprocessing.spawn(Tmp2.torch_multiprocess, (4, ), nprocs=4)