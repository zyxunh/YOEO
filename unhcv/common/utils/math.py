import numpy as np
import math


__all__ = ["point_to_line_distance"]


def point_to_line_distance(points, *, line_point=None, line_direction):
    '''
    point: N, 2
    line_point: 2
    line_direction: 2
    '''
    # 直线上的点
    if line_point is None:
        line_point = np.array([0, 0], dtype=points.dtype)
    dx, dy = line_direction

    # 法向量
    nx, ny = -dy, dx

    # 向量从直线点到每个目标点
    v = points - line_point

    # 带符号的距离：投影到法向量方向，再除以法向量模长, 正负表示在哪一侧
    signed_distances = (v @ np.array([nx, ny])) / np.hypot(nx, ny)
    return signed_distances