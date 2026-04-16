import copy
import string
from io import BytesIO
from typing import Union, List

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os

from unhcv.common.utils import write_im, get_base_name, obj_load
from PIL import ImageFont, ImageDraw, Image

from .color_map import get_idx_color

# from unhcv.common.image.visual import imwrite_tensor, visual_tensor, concat_differ_size
PATH = os.path.dirname(os.path.realpath(__file__))

# 设置字符串长度
def SetFixedStrLength(text, font, width):
    strList = []
    newStr = ''
    index = 0
    for item in text:
        newStr += item
        if font.getsize(newStr)[0] > width:
            #     print(font.getsize(newStr)[0])
            strList.append(newStr)
            newStr = ''
            # 如果后面长度不没有定长长就返回
            if font.getsize(text[index:])[0] < width + 20:
                strList.append(text[index:])
                break

        index += 1
    resStr = ''
    count = 0
    for item in strList:
        resStr += item+'\n'
        count += 1

    return resStr, count

def putTextBatch(show_texts, img_size, **kwargs):
    text_show = [putText(show_texts=var, img_size=img_size, **kwargs) for var in show_texts]
    text_show = concat_differ_size(text_show, axis=0)
    return text_show

def putText(img=None, point=None, show_texts=None, img_size=512, line_words=None, font_scale=0.1, max_str_num_in_line=23, mode="PIL"):
    if isinstance(show_texts, str):
        show_texts = [show_texts]
    if mode == "PIL":
        if img is None:
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            img_size = img_size[::-1]
            img = Image.new('RGB', img_size, (0, 0, 0, 0))
        else:
            img = Image.fromarray(img)
        font_size = max(int(img.width * font_scale), 1)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(os.path.join(PATH, 'fonts', 'Arial.ttf'), font_size)
        if line_words is None:
            line_words = max(int(img.width / font_size * 2), 1)
        import textwrap
        current_h, pad = 10, 10
        for show_text in show_texts:
            show_text_split = textwrap.wrap(show_text, width=line_words)
            h = 0
            for line in show_text_split:
                bbox = draw.textbbox((0, 0), line, font=font); w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
                draw.text((pad, current_h), line, font=font)
                current_h += h + pad
            current_h += h + pad
        return np.array(img)
    else:
        img = img.copy()
        point = list(point)
        font_size = max(img.shape[1] * font_scale, 0)
        size_height = int(30 * font_size)
        point[1] += size_height
        if not isinstance(show_texts, (list, tuple)):
            show_texts = [show_texts]
        for _text in show_texts:
            while(1):
                text = _text[:max_str_num_in_line]
                _text = _text[len(text):]
                if len(text) == 0:
                    break
                font = cv2.FONT_HERSHEY_SIMPLEX
                # paint_chinese_opencv(im, text, (0, bottom), font_size, (0, 0, 255))
                cv2.putText(img, text, point, font, font_size, (0, 0, 255), 2)
                point[1] += size_height
    return img

def visual_bbox(img, bboxes, box_type='bbox', stack_dim=0, ori_hw=None):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    img_bboxes = img.copy()
    for i in range(len(bboxes)):
        bbox_ = bboxes[i]
        if 0 and isinstance(bbox_, UnhInstances):
            bbox = bbox_.bboxes
            texts = []
            for key, value in bbox_.get_fields().items():
                if key != 'bboxes':
                    if isinstance(value, (np.float32, np.float64)):
                        value = round(float(value), 2)
                    texts.append(f'{key}: {value}')
            img_bboxes = putText(img_bboxes, bbox[:2].astype(np.int64), texts)
            cv2.rectangle(img_bboxes, tuple(bbox[:2].astype(np.int64)), tuple(bbox[2:].astype(np.int64)), color=get_idx_color(i + 1), thickness=2)
        else:
            bbox = bbox_
            if ori_hw is not None:
                bbox_image_id = bbox[0]
                bbox = bbox[1:]
                if stack_dim == 0:
                    bbox[1::2] += ori_hw[0] * bbox_image_id
                else:
                    bbox[0::2] += ori_hw[1] * bbox_image_id
            if box_type == 'bbox':
                cv2.rectangle(img_bboxes, tuple(bbox[:2].astype(np.int64)), tuple(bbox[2:].astype(np.int64)), color=get_idx_color(i + 1), thickness=2)
            elif box_type == 'polygon':
                cv2.polylines(img_bboxes, [bbox.astype(np.int64).reshape(-1, 1, 2)], True, color=get_idx_color(i + 1), thickness=2)
            else:
                raise NotImplementedError
    return img_bboxes
    # show_img = np.concatenate((img, img_bboxes), axis=0)
    # return show_img

def visual_points(img, points, thickness=1, dynamic_thickness=True, base_img_size=320, stack_axis=0, fix_color=None):
    if dynamic_thickness:
        thickness = int(min(img.shape[:2]) / base_img_size * thickness + 0.5)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points = points.astype(np.int64)
    img_draw = img.copy()
    img_shape = img.shape
    clip = lambda x, i_shpae: min(max(x, 0), img_shape[i_shpae])
    points_color = np.zeros_like(img, dtype=np.float32)
    for i in range(len(points)):
        point = points[i]
        if point[0] > img.shape[1] or point[1] > img.shape[0] or (point < 0).any():
            print(f'point: {point} exceed img border')
        img_draw_point = img_draw[point[1]-thickness:point[1]+thickness, point[0]-thickness: point[0]+thickness]
        if fix_color is None:
            color = get_idx_color(i + 1)
        else:
            color = fix_color[i]
        img_draw_point[...] = color * 0.5 + img_draw_point * 0.5
        # points_color[clip(point[1]-thickness, 1):clip(point[1]+thickness, 1), clip(point[0]-thickness, 0): clip(point[0]+thickness, 0)] = color
        points_color[point[1] - thickness: point[1] + thickness, point[0] - thickness: point[0] + thickness] = color
    show_img = np.concatenate((img_draw, points_color), axis=stack_axis)
    return show_img


def visual_tensor(tensor: torch.Tensor,
                  image_mode='bgr',
                  max_value=None,
                  min_value=None,
                  std=None,
                  mean=None,
                  square_flatten=False,
                  max_num=None,
                  reverse=False,
                  row_column=None,
                  min_size=None,
                  max_size=None,
                  stack_dim=0,
                  row_column_line_width=2,
                  rescale=True, name=None
                  ):
    if max_num is not None:
        tensor = tensor[:max_num]
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.dim() == 3:
        tensor = tensor[:, None]
    if tensor.dim() == 2:
        tensor = tensor[None]
    if std is not None:
        mean = tensor.new_tensor(mean); std = tensor.new_tensor(std)
        min_value = (0 - mean) / std
        max_value = (1 - mean) / std
    if not isinstance(max_value, torch.Tensor) and max_value is not None:
        max_value = tensor.new_tensor(max_value); min_value = tensor.new_tensor(min_value)
    if max_value is not None:
        tensor_max = max_value[..., None, None]
    else:
        tensor_max = tensor.flatten(-2, -1).max(-1)[0][..., None, None]
    if min_value is not None:
        tensor_min = min_value[..., None, None]
    else:
        tensor_min = tensor.flatten(-2, -1).min(-1)[0][..., None, None]

    if tensor.dim() == 4 and tensor_max.dim() != 4:
        tensor_max = tensor_max[None]; tensor_min = tensor_min[None]

    if rescale:
        tensor = (tensor - tensor_min) * (255 / (tensor_max - tensor_min).clamp(min=1e-4))
        tensor.clamp_(min=0, max=255)
    if min_size is not None or max_size is not None:
        tensor_dim = tensor.dim()
        if tensor_dim == 3:
            tensor = tensor[None]
        elif tensor_dim == 4:
            pass
        else:
            raise NotImplementedError
        scale = 1
        if min_size is not None:
            tensor_min_size = min(tensor.shape[2:])
            if tensor_min_size < min_size:
                scale = min_size / tensor_min_size
        else:
            tensor_max_size = max(tensor.shape[2:])
            if tensor_max_size > max_size:
                scale = max_size / tensor_max_size
        tensor = torch.nn.functional.interpolate(tensor, scale_factor=scale, mode="nearest-exact")
        if tensor_dim == 3:
            tensor = tensor[0]

    if tensor.dim() == 4:
        tensor = tensor.permute(1, 0, 2, 3)
        if square_flatten:
            num = tensor.size(1)
            num_hw = int(np.ceil(num ** 0.5))
            pad_num = num_hw ** 2 - num
            if pad_num > 0:
                tensor = torch.cat([tensor, tensor.new_zeros([tensor.size(0), pad_num, *tensor.shape[2:]])], dim=1)
            tensor = images2row_column(tensor, num_hw, num_hw)
        elif row_column is not None:
            num = tensor.size(1)
            if isinstance(row_column, int):
                column = num // row_column
                row = row_column
            else:
                row, column = row_column
            tensor = images2row_column(tensor, row, column)
        else:
            if stack_dim == 0:
                tensor = tensor.flatten(1, 2)
            elif stack_dim == 1:
                tensor = tensor.transpose(1, 2).flatten(2, 3)
    tensor = tensor.permute(1, 2, 0)

    if reverse:
        tensor = tensor.flip(-1)
    elif image_mode == 'rgb':
        tensor = tensor.flip(-1)
    if tensor.shape[-1] == 1:
        tensor.squeeze_(-1)
    show_tensor = tensor.round().detach().to(torch.uint8).cpu().numpy()
    if name is not None:
        show_tensor = putText(show_tensor, show_texts=name)
    return show_tensor


def images2row_column(tensor, row, column, row_column_line_width=5, color=255):
    tensor = tensor.reshape(tensor.size(0), row, column, *tensor.shape[2:])
    tensor = tensor.permute(0, 1, 3, 2, 4)
    tensor = torch.cat([tensor, color + tensor.new_zeros(
        [tensor.size(0), tensor.size(1), row_column_line_width, tensor.size(3), tensor.size(4)])], dim=2)
    tensor = torch.cat([tensor, color + tensor.new_zeros(
        [tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3), row_column_line_width])], dim=4)
    tensor = tensor.flatten(-2).flatten(-3, -2)
    return tensor


def concat_differ_size_tensor(imgs):
    max_size = max([var.shape[2] for var in imgs])
    scales = [max_size / var.shape[2] for var in imgs]
    imgs = [F.interpolate(img, scale_factor=scale, mode="nearest-exact") for img, scale in zip(imgs, scales)]
    imgs = torch.cat(imgs, dim=0)
    return imgs


def concat_differ_size(imgs, axis=1, max_size=None, same_all_axis=False):
    if axis == 1:
        keep_size_axis = 0
    elif axis == 0:
        keep_size_axis = 1
    else:
        raise NotImplementedError
    if same_all_axis:
        image_shape = imgs[0].shape[:2][::-1]
    imgs = [img for img in imgs if img is not None]
    keep_size = [img.shape[keep_size_axis] for img in imgs]
    max_keep_size = max(keep_size)
    if max_size is not None:
        max_keep_size = min(max_size, max_keep_size)
    ndims = [img.ndim for img in imgs]
    max_ndims = max(ndims)
    new_imgs = []
    for img in imgs:
        scale = max_keep_size/img.shape[keep_size_axis]
        if same_all_axis:
            img = cv2.resize(img, image_shape, interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if max_ndims == 3 and img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, 2)
        new_imgs.append(img)
    new_imgs = np.concatenate(new_imgs, axis=axis)
    return new_imgs


def pad_image(img: Union[np.ndarray, Image.Image], pad_hw, pad_value=255):
    if isinstance(img, Image.Image):
        img_pil_flag = True
        if img.width == pad_hw[1] and img.height == pad_hw[0]:
            return img
        img = np.array(img)
    else:
        img_pil_flag = False
        if img.shape[1] == pad_hw[1] and img.shape[0] == pad_hw[0]:
            return img
    if img.ndim == 3:
        pad_hw = (*pad_hw, img.shape[-1])
    img_pad = np.zeros(pad_hw, dtype=img.dtype) + pad_value
    img_pad[:img.shape[0], :img.shape[1]] = img
    if img_pil_flag:
        img_pad = Image.fromarray(img_pad)
    return img_pad


def pad_image_to_same_size(imgs: List[Union[np.ndarray, Image.Image]], pad_value=255):
    img_hws = []
    for img in imgs:
        if isinstance(img, Image.Image):
            img_hws.append((img.height, img.width))
        else:
            img_hws.append(img.shape[:2])
    img_hws = list(zip(*img_hws))
    img_max_hw = (max(img_hws[0]), max(img_hws[1]))
    new_imgs = []
    for img in imgs:
        new_imgs.append(pad_image(img, pad_hw=img_max_hw, pad_value=pad_value))
    return new_imgs


def imwrite_tensor(path: str, tensor: torch.Tensor, max_value=None, min_value=None, **kwargs):
    tensor_show = visual_tensor(tensor, max_value=max_value, min_value=min_value, **kwargs)
    try:
        if tensor_show.ndim == 3 and tensor_show.shape[2] > 3:
            tensor_show = tensor_show[..., :3]
        write_im(path, tensor_show)
    except Exception as ex:
        print(f"imwrite_tensor wrong!!!, tensor_show's shape {tensor_show.shape}")
        print(ex)
    return tensor_show

def resize_to_same_size(imgs):
    size = imgs[0].shape[:2]
    new_imgs = []
    for img in imgs:
        img = cv2.resize(img, size[::-1], interpolation=cv2.INTER_LINEAR)
        new_imgs.append(img)
    return new_imgs


def plt2numpy(fig=None):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # 使用 PIL 读取并转换为 NumPy 数组
    image = Image.open(buf)
    image_array = np.array(image)[..., :-1]

    if fig is None:
        fig = "all"
    plt.close(fig)
    buf.close()
    return image_array


def visual_graph_on_image(image, points, graph_matrix, colors=None):
    if colors is not None:
        colors = colors / 255
    num_points = len(points)
    assert num_points  == len(graph_matrix)

    fig, ax = plt.subplots()
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(image)

    for i in range(num_points):
        for j in range(num_points):
            graph_matrix_i_j = graph_matrix[i, j]
            if i != j and graph_matrix_i_j != 0:
                start_point = points[i]
                end_point = points[j]
                if colors is not None:
                    color = colors[i]
                else:
                    color = 'b'
                if graph_matrix_i_j == 1:
                    arrowstyle = '->'
                elif graph_matrix_i_j == 2:
                    arrowstyle = '<->'
                else:
                    raise NotImplementedError
                arrow = FancyArrowPatch(start_point, end_point, arrowstyle=arrowstyle, color=color, mutation_scale=20)
                ax.add_patch(arrow)

    plt.axis('off')
    image_array = plt2numpy(fig)
    return image_array

    # plt.savefig('/home/zhuyixing/show/test1.png', bbox_inches='tight', pad_inches=0)
    # plt.tight_layout()
    fig.tight_layout(pad=0)
    fig.subplots_adjust(0, 0, 1, 1)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    fig.canvas.draw()  # 必须先渲染图形
    breakpoint()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())[..., :-1]

    plt.close(fig)  # 关闭图形以释放内存
    return image_array


def get_mid_top_loc(poly, image_h, image_w):
    # poly: (n, 2)
    top_idx = poly[:, 1].argmin()
    mid_top = (np.clip(int(poly[top_idx][0]), 25, image_w - 35),
               np.clip(int(poly[top_idx][1]), 25, image_h - 35))
    return mid_top

def get_mid_top_from_masks(masks):
    """
    :param masks: n,h,w
    :return:
    """
    mid_tops = []
    for i in range(masks.shape[0]):
        if masks[i].sum() == 0:
            mid_tops.append([-1, -1])
            continue
        contours, _ = cv2.findContours(np.uint8(masks[i]), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # [n1,1,2], [n2,1,2]
        poly = np.concatenate(contours)[:, 0]
        mid_tops.append(get_mid_top_loc(poly, masks.shape[-2], masks.shape[-1]))
    return mid_tops

def draw_polyline_with_mask(I, mask, c):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    I = cv2.drawContours(I, contours, -1, c, 3)
    return I

def fill_poly_with_mask(I, polyfilled, mask_instance, c, alpha=0.6):
    polyfilled[np.where(mask_instance == 1)] = c
    I = cv2.addWeighted(I, alpha, polyfilled, 1 - alpha, 0)
    return I

def write_text(img, text, location, c):
    cv2.rectangle(img, (location[0] - 5, location[1] - 10), (location[0] + 25, location[1] + 5), c, -1)
    cv2.putText(img, text, location, 0, 0.35, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img

def visual_instance_mask_and_ID(I, masks, mid_tops=None, colors_rainbow=None, IDs=None):
    if mid_tops is None:
        mid_tops = get_mid_top_from_masks(masks)
    upper_lower = string.ascii_uppercase + string.ascii_lowercase + string.ascii_uppercase
    I_masked = copy.deepcopy(I)
    num_inst = len(masks) if type(masks) == list else masks.shape[0]  # masks: n,h,w
    colors = list((np.random.random(size=3 * num_inst).reshape(num_inst, -1) + 0.0001) * 255)
    if colors_rainbow != None:
        colors[:len(colors_rainbow)] = colors_rainbow

    # show all annotations
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue

        I_masked = draw_polyline_with_mask(I_masked, mask, colors[i])
        I_masked = fill_poly_with_mask(I_masked, copy.deepcopy(I_masked), mask, colors[i], alpha=0.6)

    # # another loop to put instance ID on top
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        # text = f"{str(i)}" if IDs == None else IDs[i]
        # text = IDs[i]
        text = upper_lower[i] if IDs == None else IDs[i]
        I_masked = write_text(I_masked, text, mid_tops[i], colors[i])

    return I_masked

def visual_feature(features, mode="pca", max_num=None, stack_dim=0, square_flatten=False):
    from sklearn.decomposition import PCA

    """
    将高维特征（N, D）降维到 RGB 图像（N, 3）

    Args:
        features: np.ndarray of shape (N, D) 高维特征，比如 patch embedding 或者中间层特征

    Returns:
        rgb: np.ndarray of shape (N, 3)，范围 [0, 1]
    """
    if max_num is not None:
        features = features[:max_num]
    if isinstance(features, torch.Tensor):
        features_ndim = features.ndim
        hw = features.shape[2:]
        if features_ndim == 4:
            features = features.detach().permute(0, 2, 3, 1).flatten(1, 2).cpu().float().numpy()
    else:
        raise TypeError("features should be np.ndarray or torch.Tensor")
        features_ndim = None
    pca = PCA(n_components=3)
    rgbs = []
    for feature in features:
        reduced = pca.fit_transform(feature)

        # 标准化到 0-1
        min_val = reduced.min(axis=0)
        max_val = reduced.max(axis=0)
        rgb = ((reduced - min_val) / (max_val - min_val + 1e-5) * 255).astype(np.uint8)
        if features_ndim is not None:
            rgb = rgb.reshape(*hw, -1)
            rgbs.append(rgb)
    if square_flatten:
        rgbs = np.stack(rgbs, axis=0)
        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2)
        rgbs = visual_tensor(rgbs, min_value=0, max_value=255, square_flatten=True)
    else:
        rgbs = np.concatenate(rgbs, axis=stack_dim)
    return rgbs

def masks2mask(masks, thres=0.5):
    if masks.size(0) == 0:
        return torch.zeros(masks.shape[1:], dtype=torch.int64, device=masks.device)
    tmp, mask = masks.max(0)
    mask += 1; mask[tmp < thres] = 0
    return mask


if __name__ == '__main__':
    data = obj_load("/home/zhuyixing/show/for_order_debug.pkl")
    image = data["image"][..., ::-1]
    modal = data["modal"]
    gt_depth_order = data["gt_depth_order"]
    centers = []
    for modal_i in modal:
        centers.append(np.argwhere(modal_i == 1).mean(0)[::-1])
    from .color import visual_mask, COLOR_MAP
    modal_indices, tmp = modal.argmax(0) + 1, modal.max(0); modal_indices[tmp == 0] = 0
    image_show = visual_mask(image, modal_indices, color_map=COLOR_MAP)[0].astype(np.uint8)
    image_show = visual_graph_on_image(image_show, centers, gt_depth_order, colors=COLOR_MAP[1:] / 255)
    cv2.imwrite("/home/zhuyixing/show/test1.png", image_show)
    breakpoint()
    shows = putTextBatch(['show1', 'show2'], (512, 256))
    # img = torch.zeros([10, 3, 224, 224])
    # img_show = visual_tensor(img, row_column=(2, 5))
    img = np.zeros([120, 320, 3])
    img1 = Image.fromarray(np.zeros([100, 200, 3], dtype=np.uint8))
    imgs = pad_image_to_same_size((img, img1))
    breakpoint()
    # img = np.zeros([224, 224, 3])
    # points = np.ones([10, 2], dtype=np.float32) - 2
    # show_img = visual_points(img, points, thickness=5)
    # cv2.imwrite('/home/tiger/code/unhcv/debug/test.png', show_img)
    show = putText(show_texts="The rain in Spain falls mainly on the plains. However, Pillow, the PIL fork, currently has a PR to search a Linux directory. It's not exactly clear yet which directories to search for all Linux variants, but you can see the code here and perhaps contribute to the PR:")
    write_im("/mnt/bn/zhuyixing/code/unhcv/show_out.jpg", show)
    breakpoint()

    latent = torch.randn(16, 3, 64, 64)
    out = visual_tensor(latent, square_flatten=True, max_value=(1, 1, 1), min_value=(-1, -1, -1))
    cv2.imwrite('/mnt/bn/zhuyixing/code/unhcv/debug51.jpg', out)
    breakpoint()
