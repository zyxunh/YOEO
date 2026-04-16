import re
import numpy as np
import torch


def seg_remap(seg, seg_maps):
    if isinstance(seg_maps, str):
        if re.match('less(.*)to(.*)', seg_maps) is not None:
            seg = seg.astype(np.float32)
            _, old_id, new_id = re.split('less|to', seg_maps)
            seg[seg < float(old_id)] = float(new_id)
        elif re.match('thres(.*)', seg_maps) is not None:
            _, thres = re.split('thres', seg_maps)
            seg = (seg > float(thres)).astype(seg.dtype)
    elif isinstance(seg_maps, dict):
        for old_id, new_id in seg_maps.items():
            seg[seg == old_id] = new_id
    else:
        raise NotImplementedError
    return seg

def get_seg_idxes(seg, ignore_idxes=255):
    if isinstance(seg, np.ndarray):
        seg_idxes = np.unique(seg)
        valid = (seg_idxes >= 0) & (seg_idxes < 255)
        seg_idxes = seg_idxes[valid]
    elif isinstance(seg, torch.Tensor):
        seg_idxes = torch.unique(seg)
        valid = (seg_idxes >= 0) & (seg_idxes < 255)
        seg_idxes = seg_idxes[valid]
    else:
        raise NotImplementedError
    return seg_idxes

def pad_seg(seg_lt, pad_shape=None, pad_value=0):
    out = seg_lt[0].new_zeros((len(seg_lt), *pad_shape)) + pad_value
    for i, seg in enumerate(seg_lt):
        out[i, :seg.shape[0]] = seg
    return out

def mask_iou(mask1, mask2, mode='inter/self'):
    """

    Args:
        mask1: NHW
        mask2: KHW
        mode:

    Returns:
        iou: NKHW

    """
    mask1 = (mask1 > 0)[:, None]; mask2 = (mask2 > 0)[None, :]
    inter = (mask1 & mask2).sum((2, 3))
    if mode == 'inter/self':
        iou = inter / mask1.sum((2, 3)).clamp(min=1e-4)
    else:
        raise NotImplementedError
    return iou


if __name__ == '__main__':
    re.split('less|to', 'less50to-1')
    re.match('less(.*)to(.*)', 'less50to-1')
