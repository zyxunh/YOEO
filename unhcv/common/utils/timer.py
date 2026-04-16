from time import time, sleep

import torch


class Timer:
    def __init__(self, warm_up_iter=3, synchronize=False):
        self.warm_up_iter = warm_up_iter
        self.iter_num = 0
        self.time_accumulate = 0
        self.begin_time = None
        self.synchronize = synchronize
    
    def tic(self):
        if self.synchronize:
            torch.cuda.synchronize()
        self.begin_time = time()

    def toc(self, num=1):
        if self.synchronize:
            torch.cuda.synchronize()
        if self.begin_time is None:
            self.begin_time = time()
            return
        if self.iter_num >= self.warm_up_iter:
            self.time_accumulate += (time() - self.begin_time)
        self.iter_num += num
    
    def mean(self):
        if self.iter_num > self.warm_up_iter:
            return self.time_accumulate / (self.iter_num - self.warm_up_iter)
        else:
            return 0

class TimerDict:
    def __init__(self, warm_up_iter=3, synchronize=False) -> None:
        self.time_dict = {}
        self.warm_up_iter = warm_up_iter
        self.synchronize = synchronize

    def tic(self, key):
        if key not in self.time_dict:
            self.time_dict[key] = Timer(self.warm_up_iter, synchronize=self.synchronize)
        self.time_dict[key].tic()

    def toc(self, key, **kwargs):
        if key not in self.time_dict:
            self.time_dict[key] = Timer(self.warm_up_iter, synchronize=self.synchronize)
        self.time_dict[key].toc(**kwargs)
    
    def get_mean_time(self):
        mean_time_dict = {}
        for key, timer in self.time_dict.items():
            mean_time_dict[key] = timer.mean()
        return mean_time_dict
    
if __name__ == '__main__':
    timer_dict = TimerDict()
    for _ in range(5):
        timer_dict.tic('0.1')
        sleep(0.1)
        timer_dict.toc('0.1')

        timer_dict.tic('0.2')
        sleep(0.2)
        timer_dict.toc('0.2')

        timer_dict.tic('0.3')
        sleep(0.3)
        timer_dict.toc('0.3')
    print(timer_dict.get_mean_time())
        