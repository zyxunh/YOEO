from torch.utils.data import IterableDataset
from copy import deepcopy


class FakeDataset(IterableDataset):
    def __init__(self, data):
        super(FakeDataset).__init__()
        self.data = data

    def __iter__(self):
        while True:
            yield deepcopy(self.data)
