import torch.utils.data as torch_data


class IterableDatasetWrap(torch_data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)


def wrap_torch_dataset(dataset, wrap_mode="IterableDataset"):
    if wrap_mode == "IterableDataset":
        return IterableDatasetWrap(dataset)
    else:
        raise NotImplementedError

