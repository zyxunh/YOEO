import PIL
import cv2
import numpy as np
# import shapely.geometry as shgeo
import kornia
import torch
import torch.nn.functional as F
import math

try:
    from PIL import Image
except ImportError:
    Image = None


def cal_mask_center(mask, normal=False):
    # mask: h, w
    if isinstance(mask, torch.Tensor):
        yxs = torch.nonzero(mask).float()
        if len(yxs):
            yxc = yxs.mean(0)
        else:
            yxc = mask.new_tensor([-1., -1.], dtype=torch.float)
    else:
        yxc = np.argwhere(mask > 0).mean(0)
    if normal:
        yxc[0] = yxc[0] / mask.shape[0]
        yxc[1] = yxc[1] / mask.shape[1]
    return yxc


def ratio_length2hw(original_hw=None, ratio=None, length=None):
    # w * ratio * w = length
    if ratio is None:
        ratio = original_hw[0] / original_hw[1]
    w = int(round((length / ratio) ** 0.5))
    h = int(ratio * w)
    return h, w


def area2hw(area, hw_ratio, max_stride=1, area_over_scale=1.1):
    hw_ratio = round(min(max(hw_ratio, 0.25), 4), 2)
    # max_stride * w_num * hw_ratio
    # max_stride * w_num
    w_num = max((area / max_stride ** 2 / hw_ratio) ** 0.5, 1)
    h_num = max(w_num * hw_ratio, 1)
    w_num_round = round(w_num); h_num_round = round(h_num)
    max_area = area_over_scale ** 2 * area
    h = max_stride * h_num_round
    w = max_stride * w_num_round
    # if w * h > max_area:
        # if h_num_round / int(w_num) - hw_ratio >
    return h, w


def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST_EXACT,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
}

if Image is not None:
    pillow_interp_codes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'box': Image.BOX,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING,
    }


class WarpMat:

    @staticmethod
    def scale(x, y):
        mat = np.eye(3, dtype=np.float32)
        mat[0, 0] *= x
        mat[1, 1] *= y
        return mat

    @staticmethod
    def translation(x, y):
        mat = np.eye(3, dtype=np.float32)
        mat[0, 2] = x
        mat[1, 2] = y
        return mat

    @staticmethod
    def rotate(angle, center=None):
        angle = angle / 180 * np.pi
        mat = np.eye(3, dtype=np.float32)
        mat[0, 0] = np.cos(angle)
        mat[0, 1] = -np.sin(angle)

        mat[1, 0] = np.sin(angle)
        mat[1, 1] = np.cos(angle)
        if center is not None:
            mat_trans = WarpMat.translation(-center[0], -center[1])
            mat = WarpMat.combine_mat(np.linalg.inv(mat_trans), mat, mat_trans)
        return mat

    @staticmethod
    def combine_mat(*mats):
        mats = mats[::-1]
        mat = mats[0]
        for var in mats[1:]:
            mat = np.dot(var, mat)
        return mat


def warp(img, matrix, size, interpolation, border_value=0):
    img = cv2.warpAffine(
        img,
        matrix,
        size,
        flags=cv2.INTER_NEAREST if interpolation == 'nearest' else cv2_interp_codes[interpolation] ,
        borderValue=border_value,
    )
    return img


def warp_box(box, matrix, box_type='xy4'):
    if box_type == 'ltrb':
        box_shape = box.shape
        box = box.reshape(-1, 2, 2)
        box = np.append(box,
                        np.ones([box.shape[0], 2, 1], dtype=box.dtype),
                        axis=-1)
        box = np.einsum('ab,ncb->nca', matrix, box)
        box = box.reshape(box_shape)
    elif box_type == 'xy4':
        box_shape = box.shape
        box = box.reshape(-1, 4, 2)
        box = np.append(box,
                        np.ones([box.shape[0], box.shape[1], 1],
                                dtype=box.dtype),
                        axis=-1)
        box = np.einsum('ab,ncb->nca', matrix, box)
        box = box.reshape(box_shape)
    return box


def box_type_convert(bbox, convert_type='xywh2xy4'):
    raw_shape = bbox.shape
    if convert_type == 'xywh2xy4':
        xywh = bbox.reshape(-1, 4)
        ltrb = xywh.copy()
        ltrb[:, 2:] = ltrb[:, :2] + np.maximum(ltrb[:, 2:] - 1, 0)
        xy4 = np.stack([
            ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 1], ltrb[:, 2],
            ltrb[:, 3], ltrb[:, 0], ltrb[:, 3]
        ],
                       axis=1)
        dim_last = (8, )
        out = xy4
    elif convert_type == 'ltrb2xywh':
        ltrb = bbox.reshape(-1, 4)
        xywh = ltrb.copy()
        xywh[:, 2:] = ltrb[:, 2:] - ltrb[:, :2] + 1
        dim_last = (4, )
        out = xywh
    if len(raw_shape) == 2:
        out = out.reshape(-1, *dim_last)
    else:
        out = out.reshape(*dim_last)
    return out


def inter_polygon(poly1, poly2):
    poly1 = shgeo.Polygon(poly1.reshape(-1, 2))
    poly2 = shgeo.Polygon(poly2.reshape(-1, 2))
    poly_inter = poly1.intersection(poly2)
    if len(poly_inter.exterior.coords) != 5:
        poly_inter = poly_inter.minimum_rotated_rectangle.exterior
    else:
        poly_inter = poly_inter.exterior
    coords = poly_inter.coords[:4]
    if poly_inter.is_ccw:
        pass
    else:
        coords = coords[::-1]

    return np.array(coords, dtype=np.float32)


def sort_polygon(poly, key='height'):
    poly = poly.reshape(-1, 4, 2)
    if key == 'height':
        poly_h = np.minimum(np.linalg.norm(poly[:, 0] - poly[:, 3], axis=-1),
                            np.linalg.norm(poly[:, 1] - poly[:, 2], axis=-1))
        poly_w = np.minimum(np.linalg.norm(poly[:, 0] - poly[:, 1], axis=-1),
                            np.linalg.norm(poly[:, 2] - poly[:, 3], axis=-1))
        value = np.minimum(poly_h, poly_w)
    else:
        value = np.array([shgeo.Polygon(var).area for var in poly])
    area_argsort = (-value).argsort()
    poly = poly[area_argsort]
    return poly


def resize(xys, scale, mode="xy"):
    if mode == "xy":
        return xys * scale

def resize_coordinate(bboxes, scale_h, scale_w=None):
    if scale_w is None:
        scale_w = scale_h
    bboxes = bboxes.clone()
    bboxes[:, ::2] = (bboxes[:, ::2] + 0.5) * scale_w - 0.5
    bboxes[:, 1::2] = (bboxes[:, 1::2] + 0.5) * scale_h - 0.5
    return bboxes

class Resize:
    @staticmethod
    def cal_target_size(ori_size, max_size, min_side_size, size_divisibility=None):
        scale = (min_side_size / min(ori_size))
        scale = min(max_size / max(ori_size), scale)
        tgt_size = [int(var * scale + 0.5) for var in ori_size]
        if size_divisibility is not None:
            tgt_size = [max(int(var / size_divisibility + 0.5), 1) * size_divisibility for var in tgt_size]
        return tgt_size

def unh_interpolate(x, size=None, scale_factor=None, mode='nearest', align_corners=False):
    assert align_corners == False
    assert mode == 'nearest'
    homography = torch.eye(3).view(1, 3, 3)
    if size is not None and isinstance(size, int):
        size = (size, size)
    if scale_factor is None:
        scale_factor = (size[0] / x.size(2), size[1] / x.size(3))
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)
    size = (int(scale_factor[0] * x.size(2) + 0.5), int(scale_factor[1] * x.size(3) + 0.5))
    scale_reciprocal = [1 / var for var in scale_factor]
    warp_mat_inv = WarpMat.combine_mat(WarpMat.translation(scale_reciprocal[1] * 0.5 - 0.5, scale_reciprocal[0] * 0.5 - 0.5), WarpMat.scale(scale_reciprocal[1], scale_reciprocal[0]))
    warp_mat_inv = x.new_tensor(warp_mat_inv, dtype=torch.float)[None]
    warp_mat = torch.inverse(warp_mat_inv)
    if x.dtype in [torch.long, torch.int]:
        x_ = x.float()
    else:
        x_ = x
    out = kornia.geometry.warp_affine(x_, warp_mat[:, :2], dsize=size, mode=mode, align_corners=True).to(x)
    return out

def resize_img(img, size=None, max_size=None, scale=None, mode='bilinear', stride=None):
    """
    size: w, h
    """
    if isinstance(img, np.ndarray):
        img_size = np.array(img.shape[:2])
    elif isinstance(img, PIL.Image.Image):
        img_size = np.array(img.size)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    if size is not None:
        resized_size = np.array(size)
    elif max_size is not None:
        # img_size = np.array(img.shape[:2])
        scale = max_size / max(img_size)
        resized_size = img_size * scale
    else:
        scale = np.array(scale)
        # img_size = np.array(img.shape[:2])
        resized_size = img_size * scale
    if stride is not None:
        resized_size = np.maximum((resized_size / stride + 0.5).astype(np.int64), 1) * stride
    else:
        resized_size = (resized_size).round().astype(np.int64)
    if isinstance(img, np.ndarray):
        img = cv2.resize(img, resized_size, interpolation=cv2_interp_codes[mode])
    elif isinstance(img, PIL.Image.Image):
        img = img.resize(resized_size.tolist(), resample=pillow_interp_codes[mode])
    return img

def grid_sample(img, coord: torch.Tensor, align_corners=True, mode='bilinear', padding_mode='zeros', norm_coord=False):
    # img is NxCxHxW
    # coord is NxKx2
    if norm_coord:
        coord = coord.clone()
        coord[..., 0] /= max(img.size(-1) - 1, 1)
        coord[..., 1] /= max(img.size(-2) - 1, 1)
    coord = (coord - 0.5) * 2

    coord_dim = coord.dim()
    if coord_dim == 3:
        coord = coord.unsqueeze(1)

    sample = F.grid_sample(img, coord, align_corners=align_corners, mode=mode, padding_mode=padding_mode)

    if coord_dim == 3:
        sample = sample.squeeze(2)
        sample = sample.transpose(1, 2)

    return sample

def expand_box(box, ratio, height=None, width=None):
    box_wh= box[2:] - box[:2]
    if isinstance(ratio, (int, float)):
        ratio_lt = ratio_rb = ratio
    else:
        if len(ratio) == 4:
            ratio_lt = ratio[:2]
            ratio_rb = ratio[2:]
        else:
            ratio_lt = ratio_rb = ratio

    if isinstance(box, np.ndarray):
        box_expanded = np.concatenate((box[:2] - box_wh * ratio_lt, box[2:] + box_wh * ratio_rb))
        if height is not None:
            box_expanded = np.maximum(box_expanded, 0)
            box_expanded[1::2] = np.minimum(box_expanded[1::2], height)
            box_expanded[::2] = np.minimum(box_expanded[::2], width)
    elif isinstance(box, torch.Tensor):
        box_expanded = torch.cat((box[:2] - box_wh * ratio_lt, box[2:] + box_wh * ratio_rb))
        if height is not None:
            box_expanded[1::2] = box_expanded[1::2].clamp(min=0, max=height)
            box_expanded[0::2] = box_expanded[0::2].clamp(min=0, max=width)

    # box_expanded[:2] -= box_wh * ratio_lt
    # box_expanded[2:] += box_wh * ratio_rb

    return box_expanded

def random_from_box(box: np.ndarray, ratio_in: float, ratio_out: float=None, box_out=None, ratio_w2h=None, img_shape=None):
    # box_wh= box[2:] - box[:2]
    # if box_out is not None:
    #     box_out_wh = box_out[2:] - box_out[:2]
    if ratio_out is None:
        ratio_out = ratio_in
    if not isinstance(ratio_in, (int, float)):
        ratio_in = np.array(ratio_in, dtype=np.float32)
    if not isinstance(ratio_out, (int, float)):
        ratio_out = np.array(ratio_out, dtype=np.float32)

    if box_out is None:
        box_out = box
    
    box_expanded = expand_box(box, -ratio_in)
    xy2_min = box_expanded[2:]
    xy1_max = box_expanded[:2]
    # xy2_min = box[2:] - box_wh * ratio_in_lt
    # xy1_max = box[:2] + box_wh * ratio_in_rb
    box_out_expanded = expand_box(box_out, ratio_out)
    xy1_min = box_out_expanded[:2]
    xy2_max = box_out_expanded[2:]
    # if box_out is not None:
    #     xy1_min = box_out[:2] - box_out_wh * ratio_out
    #     xy2_max = box_out[2:] + box_out_wh * ratio_out
    # else:
    #     xy1_min = box[:2] - box_wh * ratio_out
    #     xy2_max = box[2:] + box_wh * ratio_out
    xy1_max = np.maximum(xy1_max, xy1_min)
    xy2_min = np.maximum(xy2_min, xy1_max + 1)
    xy2_max = np.maximum(xy2_max, xy2_min)
    
    # print('xy1_min', xy1_min)
    # print('xy1_max', xy1_max)
    # print('xy2_min', xy2_min)
    # print('xy2_max', xy2_max)

    x1 = np.random.uniform(xy1_min[0], xy1_max[0])
    y1 = np.random.uniform(xy1_min[1], xy1_max[1])
    x2 = np.random.uniform(xy2_min[0], xy2_max[0])
    y2 = np.random.uniform(xy2_min[1], xy2_max[1])
    
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def keep_box_ratio(box: np.ndarray, max_ratio=None):
    box_wh = box[2:] - box[:2]
    box_center = (box[2:] + box[:2]) / 2
    if box_wh[0] > box_wh[1]:
        box_wh[1] = max(box_wh[1], box_wh[0] / max_ratio)
        # box_wh[0] = min(box_wh[0], box_wh[1] * max_ratio)
    else:
        box_wh[0] = max(box_wh[0], box_wh[1] / max_ratio)
        # box_wh[1] = min(box_wh[1], box_wh[0] * max_ratio)
    _box = box.copy()
    _box[:2] = box_center - box_wh * 0.5
    _box[2:] = box_center + box_wh * 0.5
    return _box

def box_contain_2box(box1, box2):
    box = box1.copy()
    box[:2] = np.minimum(box1[:2], box2[:2])
    box[2:] = np.maximum(box1[2:], box2[2:])
    return box

def cal_target_size(img_shape, max_size, resize_type='max_side', stride=1):
    if resize_type == 'max_side':
        scale = max_size / max(img_shape)
        tgt_shape = tuple(map(lambda x: x * scale, img_shape))
        tgt_shape = tuple(map(lambda x: max(round(x/stride), 1) * stride, tgt_shape))
        # tgt_shape = list(tgt_shape)
    
    return tgt_shape


def masks2bboxes(masks: torch.Tensor):
    """
    将多个 mask 转换为 bounding box。

    参数:
        masks (torch.Tensor): 二值化 mask 的张量，形状为 (N, H, W)，
                              其中 N 是 mask 数量，H 和 W 是 mask 的高度和宽度。

    返回:
        torch.Tensor: 每个 mask 的 bounding box，形状为 (N, 4)，格式为 [x_min, y_min, x_max, y_max]。
    """
    if masks.dim() != 3:
        raise ValueError("masks 需要是一个三维张量，形状为 (N, H, W)")

    N, H, W = masks.shape
    bboxes = []

    for i in range(N):
        mask = masks[i]
        nonzero = torch.nonzero(mask, as_tuple=True)  # 获取非零像素的索引
        if len(nonzero[0]) == 0:
            # 如果 mask 为空，返回一个空框
            bboxes.append(masks.new_tensor([0, 0, 0, 0], dtype=torch.float32))
        else:
            y_min, y_max = nonzero[0].min(), nonzero[0].max()
            x_min, x_max = nonzero[1].min(), nonzero[1].max()
            bboxes.append(torch.stack([x_min, y_min, x_max, y_max]).to(dtype=torch.float32))

    return torch.stack(bboxes)

if __name__ == '__main__':
    for hw_ratio in np.arange(0.2, 1.0001, 0.01):
        hw = area2hw(512 * 512, hw_ratio, 64)
        print(hw_ratio, hw[0] * hw[1], hw)
    breakpoint()
    # poly1 = np.array([0, 0, 100, 0, 100, 100, 0, 100], dtype=np.float32)
    # poly2 = np.array([-1, 50, 50, -1, 101, 50, 50, 101], dtype=np.float32)
    # poly_inter = inter_polygon(poly1, poly2)
    # import torch.nn.functional as F
    # x = torch.arange(21, dtype=torch.float)
    # x = x.view(1, 1, 1, 21)
    # out1 = F.interpolate(x, scale_factor=(1, 2.45), align_corners=False, mode='bilinear')
    # out2 = F.interpolate(x, scale_factor=(1, 0.25), mode='bilinear')
    # x_out = interpolate(x, scale_factor=(1, 2.45), align_corners=False, mode='nearest')
    # x = torch.arange(21, dtype=torch.float)
    # x = x.view(1, 1, 21, 1)
    # x = torch.concat((x, x), dim=1)
    # coord = torch.arange(21, dtype=torch.float) * 0.5
    # coord = torch.stack([torch.zeros_like(coord), coord], dim=-1)

    # sample = grid_sample(x, coord[None], norm_coord=True, align_corners=True)

    tgt_shape = cal_target_size(np.array([108, 171]), 512, stride=64)
    breakpoint()

    box = np.array([0, 2, 12, 10])
    print(random_from_box(box, 0.0, 1))

    breakpoint()