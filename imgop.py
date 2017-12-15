import numpy as np
import math


def gauss(x, u=0.0, sigma=1.0):
    """
    高斯函数
    :param x: 随机变量
    :param u: 期望值
    :param sigma: 标准差
    :return: 随机变量对应高斯分布的概率值
    """
    if sigma == 0:
        sigma = 1e-3
    if type(x) == np.ndarray:
        return np.exp(-np.multiply(x - u, x - u) * 0.5 / sigma / sigma) / (sigma * math.sqrt(2 * math.pi))
    else:
        return math.exp(-(x - u) * (x - u) * 0.5 / sigma / sigma) / (sigma * math.sqrt(2 * math.pi))


def conv2d(x: np.ndarray, y: np.ndarray, deep=None):
    shape = x.shape
    if deep is not None:
        x = x.reshape((-1, deep))
        y = y.reshape((-1, deep))
    else:
        x = x.reshape(-1)
        y = y.reshape(-1)
    return np.sum(np.multiply(x, y), axis=0)


def get_pixel(image: np.ndarray, row, col):
    width, height = image.shape[0], image.shape[1]
    row = 0 if row < 0 else row if row < width else width - 1
    col = 0 if col < 0 else col if col < height else height - 1
    return image[row, col]


def get_subrect(image: np.ndarray, row, col, height, width):
    """
    提取图像的子区域
    :param image: 原图像素矩阵
    :param row: 行坐标
    :param col: 列坐标
    :param height: 区域高度
    :param width: 区域宽度
    :return: 图像子区域像素矩阵
    """
    return np.array(
        [
            [
                get_pixel(image, row + tr, col + tc)
                for tc in range(width)
                ]
            for tr in range(height)
            ],
        np.int32
    )