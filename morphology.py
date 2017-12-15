# -*- coding: utf-8 -*-
import numpy as np
import imgop as iop

def corrode(image:np.ndarray, mask:np.ndarray=np.ones((3, 3))):
    """
    腐蚀操作
    :param image: 原图（二值图）
    :param mask: 结构化像素mask
    :return: 腐蚀变换后的图
    """
    result = np.zeros(image.shape)
    for n in range(image.shape[0]):
        for m in range(image.shape[1]):
            subrect = iop.get_subrect(image, n, m, mask.shape[0], mask.shape[1])
            if np.sum(np.bitwise_and(subrect == 0, mask == 1)) > 0:
                result[n,m] = 0     # set black
            else:
                result[n, m] = image[n, m]
    return result


def expand(image:np.ndarray, mask:np.ndarray=np.ones((3, 3))):
    """
    膨胀操作
    :param image: 原图（二值图）
    :param mask: 结构化像素mask
    :return: 膨胀变换后的图
    """
    result = np.zeros(image.shape)
    for n in range(image.shape[0]):
        for m in range(image.shape[1]):
            subrect = iop.get_subrect(image, n, m, mask.shape[0], mask.shape[1])
            if np.sum(np.bitwise_and(subrect != 0, mask == 1)) > 0:
                result[n, m] = 255   # set white
            else:
                result[n, m] = image[n, m]
    return result


def open(image:np.ndarray, mask:np.ndarray=np.ones((3, 3))):
    """
    开操作
    用于去除其它不符合结果元素的前景区域像素
    :param image: 原图（二值图）
    :param mask: 结构化像素mask
    :return: 开操作后的图
    """
    return expand(corrode(image, mask), mask)


def close(image:np.ndarray, mask:np.ndarray=np.ones((3, 3))):
    """
    闭操作
    用于保留背景区域与结构元素形状相似的像素，去掉其它不符合的背景区域
    :param image: 原图（二值图）
    :param mask: 结构化像素mask
    :return: 开操作后的图
    """
    return corrode(expand(image, mask), mask)




# if __name__ == '__main__':
#     from scipy.misc import imread, imshow, imsave
#     from edge import binaryscale, grayscale
#     im = imread("1.jpg")
#     im = grayscale(im)
#     im = binaryscale(im)
#     imsave('binary.jpg', im, "JPEG")
#     im = open(im, np.ones((10,10)))
#     im = open(im, np.ones((10,10)))
#     im = open(im, np.ones((10,10)))
#     # imshow(im)
#     imsave("open.jpg", im, "JPEG")