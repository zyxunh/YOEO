import torch
import torch.nn.functional as F


class OHEMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loss_for_sort, /, *, loss=None, sample_num):
        values, indices = torch.topk(loss_for_sort, sample_num)
        return indices

    @torch.no_grad()
    def sample(self, loss_for_sort: torch.Tensor, *, valid_mask=None, sample_num, deleted_indices=False, return_mask=False):
        if valid_mask is None:
            valid_mask = torch.ones_like(loss_for_sort, dtype=torch.bool)
        loss_for_sort = loss_for_sort.clone()
        valid_num = valid_mask.sum()
        sample_num = min(valid_num, sample_num)
        if deleted_indices:
            sample_num = valid_num - sample_num
            loss_for_sort.masked_fill_(~valid_mask, torch.inf)
            _, indices = torch.topk(loss_for_sort, sample_num, largest=False)
        else:
            loss_for_sort.masked_fill_(~valid_mask, -torch.inf)
            _, indices = torch.topk(loss_for_sort, sample_num)
        if return_mask:
            mask = torch.zeros_like(loss_for_sort, dtype=torch.bool)
            mask[indices] = True
            return mask
        return indices

