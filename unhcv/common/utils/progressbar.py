import sys
from datetime import timedelta

from .format import dict2strs
from .timer import Timer
import time
from tqdm.auto import tqdm


class MeanCache:
    def __init__(self):
        self.cache = {}
        self.cache_num = {}

    def update(self, data_dict):
        for key, value in data_dict.items():
            mem = self.cache.get(key, 0)
            num = self.cache_num.get(key, 0)
            self.cache[key] = mem + value
            self.cache_num[key] = num + 1

    def clear(self):
        self.cache.clear()
        self.cache_num.clear()

    def mean(self, clear=True):
        mean_dict = {}
        for key in self.cache.keys():
            mean_dict[key] = self.cache[key] / self.cache_num[key]
        if clear:
            self.clear()
        return mean_dict


class ProgressBarTqdm:
    """A progress bar which can print the progress."""
    def __init__(self, num_step, disable=False, smoothing=0.3, **kwargs):
        self.progress_bar = tqdm(range(0, num_step), disable=disable, smoothing=smoothing, **kwargs)
        self.progress_bar.set_description("Steps")

    def update(self, num=1):
        self.progress_bar.update(num)
    
    def log(self, items: dict):
        self.progress_bar.set_postfix(**items)

class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num, *args, pre_str=None, bar_width=50, unit=60, start=True, display_gap=0.5,
                 file=None, mean_log_reset_step=0, disable=False, smoothing=None, display_gap_final=30,
                 display_gap_final_elapse=30, visual_gap=1, visual_gap_final=60, visual_gap_final_elapse=None, name=None, **kwargs):
        self.disable = disable
        if mean_log_reset_step == 0:
            mean_log_reset_step = max(task_num // 10, 1)
        self.mean_log_reset_step = mean_log_reset_step
        self.display_gap_final = display_gap_final * unit
        self.display_gap_final_elapse = display_gap_final_elapse * unit
        if visual_gap_final_elapse is None:
            visual_gap_final_elapse = display_gap_final_elapse
        self.visual_gap_final_elapse = visual_gap_final_elapse * unit
        self.visual_gap = visual_gap * unit
        self.visual_gap_final = visual_gap_final * unit
        if start:
            self.start()
        if self.disable:
            return
        self.display_gap = display_gap * unit
        self.unit = unit
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        if file is not None:
            file = (open(file, 'w'),)
        else:
            file = ()
        self.file = file + (sys.stdout, )
        self.pre_str = pre_str
        self.mean_cache = MeanCache()

        self.smoothing = smoothing
        self.name = name

    def write(self, msg):
        for file in self.file:
            file.write(msg)
            file.flush()

    # def __del__(self):
    #     for file in self.file:
    #         file.close()

    def start(self):
        # if self.task_num > 0:
        #     self.file.write(f'0/{self.task_num}, '
        #                     'elapsed: 0s, ETA:')
        # else:
        #     self.file.write('completed: 0, elapsed: 0s')
        # self.file.flush()
        # self.timer = Timer()
        self.begin_time = time.time()
        # self.timer.tic()
        self.display_time = 0 # time.time()
        self.visual_time = 0 # time.time()

    @property
    def is_visual_iter(self):
        now = time.time()
        elapsed = time.time() - self.begin_time
        visual_gap = self.visual_gap if elapsed <= self.visual_gap_final_elapse else self.visual_gap_final
        flag = now - self.visual_time > visual_gap
        if flag:
            self.visual_time = now
        return flag

    def update(self, num_tasks=1):
        if self.disable:
            return
        assert num_tasks > 0
        self.completed += num_tasks
        now = time.time()
        elapsed = now - self.begin_time
        display_gap = self.display_gap if elapsed <= self.display_gap_final_elapse else self.display_gap_final
        if now - self.display_time > display_gap or self.completed == self.task_num:
            self.display_time = now
            if elapsed > 0:
                fps = self.completed / elapsed
            else:
                fps = float('inf')
            if self.task_num > 0:
                msg = self.log_cache(elapsed, fps)
                # bar_width = min(self.bar_width,
                #                 int(self.terminal_width - len(msg)) + 2,
                #                 int(self.terminal_width * 0.6))
                # bar_width = max(2, bar_width)
                # mark_width = int(bar_width * percentage)
                # bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
                self.write(msg)
            else:
                self.write(
                    f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                    f' {fps:.1f} tasks/s')

    def log_cache(self, elapsed=None, fps=None):
        msg = ""
        if elapsed is not None:
            elapsed_str = timedelta(seconds=int(elapsed)).__str__()
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            eta_str = timedelta(seconds=eta).__str__()
            if self.name is not None:
                msg += "{}\n".format(self.name)
            msg += f'{self.completed}/{self.task_num}, ' \
                f'{fps:.1f} task/s, elapsed: {elapsed_str}, ' \
                f'ETA: {eta_str}'
            if self.pre_str is not None:
                msg = f'{self.pre_str}: ' + msg
        msg += "\n"
        msg += '{}'.format(dict2strs(self.mean_cache.mean(clear=False), 3, return_list=False))
        msg += "\n\n"
        return msg

    def log(self, _log_dict=None, /, **kwargs):
        if self.disable:
            return
        if _log_dict is None:
            _log_dict = kwargs
        if self.mean_log_reset_step is not None and self.completed % self.mean_log_reset_step == 0:
            self.write(f"\nmean_cache clear at {self.completed}\n")
            msg = self.log_cache()
            self.write(msg)
            self.mean_cache.clear()

        self.mean_cache.update(_log_dict)

    def log_at_once(self, _log_dict=None, /, **kwargs):
        if self.disable:
            return
        if _log_dict is None:
            _log_dict = kwargs

        msg = "\n"
        msg += '{}'.format(dict2strs(_log_dict, 3, return_list=False))
        msg += "\n\n"

        self.write(msg)

# class ProgressBar_(MMCVProgressBar):
#     """A progress bar which can print the progress."""

#     def __init__(self, task_num, *args, bar_width=50, start=True, display_gap=30, **kwargs):
#         super().__init__(*args, task_num=task_num, bar_width=bar_width, start=start, **kwargs)
#         self.display_gap = display_gap

#     def start(self):
#         # if self.task_num > 0:
#         #     self.file.write(f'0/{self.task_num}, '
#         #                     'elapsed: 0s, ETA:')
#         # else:
#         #     self.file.write('completed: 0, elapsed: 0s')
#         self.file.flush()
#         self.timer = Timer()
#         self.display_time = time.time()

#     def update(self, num_tasks=1):
#         assert num_tasks > 0
#         self.completed += num_tasks
#         if time.time() - self.display_time > self.display_gap or self.completed == self.task_num:
#             self.display_time = time.time()
#             elapsed = self.timer.since_start()
#             if elapsed > 0:
#                 fps = self.completed / elapsed
#             else:
#                 fps = float('inf')
#             if self.task_num > 0:
#                 percentage = self.completed / float(self.task_num)
#                 eta = int(elapsed * (1 - percentage) / percentage + 0.5)
#                 msg = f'{self.completed}/{self.task_num}, ' \
#                     f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
#                     f'ETA: {eta:5}s\n'

#                 # bar_width = min(self.bar_width,
#                 #                 int(self.terminal_width - len(msg)) + 2,
#                 #                 int(self.terminal_width * 0.6))
#                 # bar_width = max(2, bar_width)
#                 # mark_width = int(bar_width * percentage)
#                 # bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
#                 self.file.write(msg)
#             else:
#                 self.file.write(
#                     f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
#                     f' {fps:.1f} tasks/s')
#             self.file.flush()


if __name__ == '__main__':
    import time
    import numpy as np
    import torch
    if 0:
        from time import sleep
        from tqdm import tqdm
        import random

        # Default smoothing of 0.3 - irregular updates and medium-useful ETA
        for i in tqdm(range(100), smoothing=0.0):
            sleep(random.randint(0,5)/10)

        # Immediate updates - not useful for irregular updates
        for i in tqdm(range(100), smoothing=1):
            sleep(random.randint(0,5)/10)

        # Global smoothing - most useful ETA in this scenario
        for i in tqdm(range(100), smoothing=0):
            sleep(random.randint(0,5)/10)
    if 0:
        progress_bar = ProgressBarTqdm(100)
        for i in range(10):
            time.sleep(2)
            progress_bar.update()
            progress_bar.log(dict(loss=torch.tensor(0.2152112).item(), grad=i))
    if 1:
        progress_bar = ProgressBar(20, display_gap=2, file="/home/yixing/train_outputs/show/hiding/test.txt",
                                   pre_str='unhcv', unit=1, mean_log_reset_step=5, display_gap_final=3,
                                   display_gap_final_elapse=10, visual_gap_final=5, visual_gap=3, disable=True)
        for i in range(20):
            time.sleep(1)
            if progress_bar.is_visual_iter:
                print('visual')
            progress_bar.log(dict(i=np.random.randint(10000000), j=np.random.randint(10000000)))
            progress_bar.update()