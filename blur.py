"""
模糊算法汇总
模糊算法包括：盒子模糊、高斯模糊、高斯双边模糊等
正在更新...
"""
from imgop import *


def gauss_blur(image: np.ndarray, kernel_size: int = 5, pos_sigma=50.0, pixel_sigma=50.0):
    """
    高斯双边模糊
    :param image: 原图
    :param kernel_size: 核大小（直径）
    :param pos_sigma: 坐标距离标准差
    :param pixel_sigma: 像素距离标准差
    :return: 模糊后的图片
    """
    result = np.zeros(image.shape)
    radius = int(kernel_size / 2)
    # 基于坐标距离的高斯核
    dist_kernel = np.array(
        [[[math.sqrt(n * n + m * m) for x in range(3)] for m in range(-radius, radius + 1)]
         for n in range(-radius, radius + 1)])
    dist_kernel = gauss(dist_kernel, sigma=pos_sigma)
    pixel_gauss_table = [gauss(x, sigma=pixel_sigma) for x in range(0, 256)]

    def gauss_from_table(matrix: np.ndarray):
        shape = matrix.shape
        matrix = matrix.reshape((-1)).astype(np.float32)
        for i in range(matrix.shape[0]):
            matrix[i] = pixel_gauss_table[int(matrix[i])]
        return matrix.reshape(shape)

    # 高斯双边模糊操作
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            subrect = get_subrect(image, row, col, kernel_size, kernel_size)
            pixel_kernel = np.abs(subrect - subrect[radius, radius])  # 子区域中所有像素与中心点像素值做差
            pixel_kernel = gauss_from_table(pixel_kernel)  # 基于像素距离的高斯核
            kernel = np.multiply(dist_kernel, pixel_kernel)
            sum = np.sum(kernel, axis=(0, 1))
            sum[sum == 0] = 1e-3
            kernel = np.multiply(kernel, 1 / sum)  # 归一化
            result[row, col] = conv2d(subrect, kernel, 3)

    return result.astype(np.uint8)