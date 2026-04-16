# Author: zhuyixing zhuyixing@bytedance.com
# Date: 2023-02-22 03:12:47
# LastEditors: zhuyixing zhuyixing@bytedance.com
# Description:
import random

import numpy as np


__all__ = ["random_choice", "uniform", "RandomChoiceFlag", "random_choice_indices", "set_seed"]

import torch


def random_choice(x, p=None):
    return x[np.random.choice(np.arange(len(x)), p=p)]


def uniform(probs):
    if isinstance(probs[0], (list, tuple)):
        if len(probs[0]) == 3:
            prob = random_choice(probs, p=[var[2] for var in probs])[:2]
        else:
            prob = random_choice(probs)
    else:
        prob = probs

    return np.random.uniform(*prob)


class RandomChoiceFlag:
    def __init__(self, probabilities):
        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        self.probabilities_cumsum = np.cumsum(probabilities)
        self.random_value = np.random.random()
        self.i = -1

    def __call__(self):
        self.i += 1
        return self.random_value < self.probabilities_cumsum[self.i]


def random_choice_indices(num, probs, device='cuda'):
    random_number = torch.rand(num, device=device)
    choice_masks = []
    prob_now = 0
    for prob in probs:
        prob_after = prob_now + prob
        choice_mask = (random_number >= prob_now) & (random_number < prob_after)
        prob_now = prob_after
        choice_masks.append(choice_mask)
    return choice_masks


def set_seed(seed=1234, set_torch=True):
    random.seed(seed)
    np.random.seed(seed)
    if set_torch:
        torch.manual_seed(seed)

if __name__ == '__main__':
    choice_masks_ = random_choice_indices(10000, [0.05, 0.05, 0.05])
    pass
