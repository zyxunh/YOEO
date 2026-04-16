import torch
import torch.nn as nn
import math
import itertools
import numpy as np


__all__ = ["DiffJPEG", "Quantization"]


class y_dequantize(nn.Module):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width
    """

    def __init__(self, factor=1):
        super(y_dequantize, self).__init__()
        self.y_table = build_y_table()
        self.factor = factor

    def forward(self, image, factor):
        if factor is None:
            factor = self.factor
        return image * (self.y_table * factor)


class c_dequantize(nn.Module):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width
    """

    def __init__(self, factor=1):
        super(c_dequantize, self).__init__()
        self.factor = factor
        self.c_table = build_c_table()

    def forward(self, image, factor):
        if factor is None:
            factor = self.factor
        return image * (self.c_table * factor)


class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(block_merging, self).__init__()

    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        # print(patches.shape) # (1,1024,8,8)
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input:
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)

        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()

        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        # result = torch.from_numpy(result)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)


class decompress_jpeg(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """

    # def __init__(self, height, width, rounding=torch.round, factor=1):
    def __init__(self, rounding=torch.round, factor=1):
        super(decompress_jpeg, self).__init__()
        self.c_dequantize = c_dequantize(factor=factor)
        self.y_dequantize = y_dequantize(factor=factor)
        self.idct = idct_8x8()
        self.merging = block_merging()
        # comment this line if no subsampling
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()

        # self.height, self.width = height, width

    def forward(self, y, cb, cr, height, width, factor):
        components = {'y': y, 'cb': cb, 'cr': cr}
        # height = y.shape[0]
        # width = y.shape[1]
        self.height = height
        self.width = width
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k], factor)
                # comment this line if no subsampling
                height, width = int(self.height / 2), int(self.width / 2)
                # height, width = int(self.height), int(self.width)

            else:
                comp = self.y_dequantize(components[k], factor)
                # comment this line if no subsampling
                height, width = self.height, self.width
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        # comment this line if no subsampling
        image = self.chroma(components['y'], components['cb'], components['cr'])
        # image = torch.cat([components['y'].unsqueeze(3), components['cb'].unsqueeze(3), components['cr'].unsqueeze(3)], dim=3)
        image = self.colors(image)

        image = torch.min(255 * torch.ones_like(image),
                          torch.max(torch.zeros_like(image), image))
        return image / 255


class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr

    """

    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        #
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        #    result = torch.from_numpy(result)
        result.view(image.shape)
        return result


class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """

    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output:
        patch(tensor):  batch x h*w/64 x h x w
    """

    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]
        # print(height, width)
        batch_size = image.shape[0]
        # print(image.shape)
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """

    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = build_y_table()

    def forward(self, image, factor):
        if factor is None:
            factor = self.factor
        image = image.float() / (self.y_table * factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.c_table = build_c_table()

    def forward(self, image, factor):
        if factor is None:
            factor = self.factor
        image = image.float() / (self.c_table * factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """

    def __init__(self, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            # comment this line if no subsampling
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )
        self.c_quantize = c_quantize(rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(rounding=rounding, factor=factor)

    def forward(self, image, factor):
        y, cb, cr = self.l1(image * 255)  # modify

        # y, cb, cr = result[:,:,:,0], result[:,:,:,1], result[:,:,:,2]
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            # print(comp.shape)
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp, factor)
            else:
                comp = self.y_quantize(comp, factor)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


def build_y_table():
    y_table = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                            55], [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                            77], [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
        dtype=np.float32).T

    y_table = nn.Parameter(torch.from_numpy(y_table))
    return y_table

def build_c_table():
    c_table = np.empty((8, 8), dtype=np.float32)
    c_table.fill(99)
    c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                [24, 26, 56, 99], [47, 66, 99, 99]]).T
    c_table = nn.Parameter(torch.from_numpy(c_table))
    return c_table


def diff_round_back(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3



def diff_round(input_tensor):
    test = 0
    for n in range(1, 10):
        test += math.pow(-1, n+1) / n * torch.sin(2 * math.pi * n * input_tensor)
    final_tensor = input_tensor - 1 / math.pi * test
    return final_tensor


class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.


class DiffJPEG(nn.Module):
    def __init__(self, differentiable=True, quality=75):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image height
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
            # rounding = Quantization()
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        # self.decompress = decompress_jpeg(height, width, rounding=rounding,
        #                                   factor=factor)
        self.decompress = decompress_jpeg(rounding=rounding, factor=factor)

    def forward(self, x, quality=None):
        '''
        '''
        if quality is not None:
            factor = quality_to_factor(quality)
        else:
            factor = None
        org_height = x.shape[2]
        org_width = x.shape[3]
        y, cb, cr = self.compress(x, factor=factor)

        recovered = self.decompress(y, cb, cr, org_height, org_width, factor=factor)
        return recovered


if __name__ == '__main__':
    breakpoint()
    # Test code
    import cv2
    from unhcv.common.image import visual_tensor
    image = cv2.imread("/home/yixing/dataset/coco/val2017/000000147498.jpg")
    quality = 20
    image = cv2.resize(image, (640, 640))
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
    image.requires_grad_(True)
    diff_jpeg = DiffJPEG(quality=100)
    # diff_jpeg = DiffJPEG(quality=quality)
    y = diff_jpeg(image, quality=20)
    image = visual_tensor(y, max_value=1, min_value=0)
    image_before = cv2.imread(f"/home/yixing/train_outputs/test_{quality}.png", image)
    cv2.imwrite(f"/home/yixing/train_outputs/new_test_{quality}.png", image)