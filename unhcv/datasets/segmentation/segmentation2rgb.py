import numpy as np
import torch


def generate_color():
    # -1 -0.5 0.0 0.5 1
    # -1 -0.5 0.0 0.5 1
    # -1 -0.5 0.0 0.5 1
    # 125-1 = 124, 平均分成100, 10*10=100
    candidate_index = [-1.0, -0.25, -0.5, 0.0, 0.25, 0.5, 1.0]
    cand_num = len(candidate_index)
    colors_loc = []
    for q1 in range(cand_num):
        for q2 in range(cand_num):
            for q3 in range(cand_num):
                colors_loc.append([candidate_index[q1], candidate_index[q2], candidate_index[q3]])
    return np.array(colors_loc[1:], dtype=np.float32)


class Entity2Rgb:
    def __init__(self, grid_num):
        self.colors_loc = generate_color()
        self.grid_num = grid_num
        self.grid_size = 1 / grid_num
        assert self.grid_num < len(self.colors_loc)

    def get_color(self, yc, xc):
        # yc, xc 0-1
        y_bin = int(yc / self.grid_size)
        x_bin = int(xc / self.grid_size)
        color_index = int(y_bin * self.grid_num + x_bin)
        return self.colors_loc[color_index]

    def mask2rgb(self, mask, dtype=torch.float):
        mask_rgb = mask.new_zeros([*mask.shape[:2], 3], dtype=dtype)
        mask_ids = torch.unique(mask)
        mask_ids = mask_ids[mask_ids != 0]
        height, width = mask.shape
        for mask_id in mask_ids:
            mask_i = mask_id == mask
            ys, xs = torch.where(mask_i)
            yc, xc = ys.to(dtype).mean() / height, xs.to(dtype).mean() / width
            mask_rgb[mask_i] = torch.from_numpy(self.get_color(yc, xc)).to(mask_rgb)
        return mask_rgb


if __name__ == '__main__':
    entity_rgb = Entity2Rgb(16)
    mask = torch.zeros(10, 10).cuda()
    mask[0] = 1
    mask[1] = 2
    mask_rgb = entity_rgb.mask2rgb(mask)