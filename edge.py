"""
边缘提取算法汇总
这里包括：Robot、一阶导、拉普拉斯、Canny算法
"""
import blur
from imgop import *


def clamp(x):
    return 0 if x < 0 else 255 if x > 255 else x


def grayscale(image: np.ndarray):
    return np.array([
        [image[n, m][0] * 0.299 + image[n, m][1] * 0.587 + image[n, m][2] * 0.114
         for m in range(image.shape[1])]
        for n in range(image.shape[0])
    ])


def binaryscale(image: np.ndarray):
    """
    二值化图像
    :param image: 原图（灰度图）
    :return: 二值化后的图像
    """
    image = image.astype(np.float32)
    t = np.average(image)
    if t == 0: return image
    while True:
        t1, t2 = np.average(image[image >= t]), np.average(image[image < t])
        if math.fabs(t - (t1 + t2) / 2) <= 1e-3:
            break
        t = (t1 + t2) / 2
    image[image >= t] = 255
    image[image < t] = 0
    return image


def laplacian(image: np.ndarray, radius: int = 1):
    """
    拉普拉斯边缘提取算法
    :param image: 原图
    :return: 边缘提取后的二值图
    """
    # 生成拉普拉斯银子
    ly = np.array([[-1 if m == radius else 0 for m in range(radius * 2 + 1)] for n in range(radius)])
    lx = np.array([-1 if m != radius else 4 * radius for m in range(radius * 2 + 1)])
    operator = np.row_stack((ly, lx, ly))
    kernel_size = radius * 2 + 1

    # 生成灰度图
    image = grayscale(image)

    result = np.zeros(image.shape)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            subrect = get_subrect(image, row, col, kernel_size, kernel_size)
            result[row, col] = clamp(conv2d(subrect, operator, None))

    return binaryscale(result.astype(np.uint8))


def canny(image: np.ndarray, tl, th):
    """
    Canny边缘提取算法（基于Sobel算子）
    :param image: 原图
    :return: 边缘提取后的二值图
    """
    # 生成灰度图
    image = grayscale(image).astype(np.float32)

    gx = np.zeros(image.shape, np.float32)  # X方向梯度
    gy = np.zeros(image.shape, np.float32)  # Y方向梯度
    amplitude = np.zeros(image.shape, np.float32)  # 变化幅度
    angle = np.zeros(image.shape, np.float32)  # 变化角度
    result = np.zeros(image.shape, np.float32)  # 最终结果
    op_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])
    op_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ])

    def solve_angle(gx, gy):
        if gx == 0:
            return 90
        elif gy == 0:
            return 0
        else:
            angle = math.atan(gy / gx) * 180 / math.pi
            return angle if angle >= 0 else angle + 180

    # 梯度计算
    print("梯度计算")
    for n in range(image.shape[0]):
        for m in range(image.shape[1]):
            subrect = get_subrect(image, n, m, 3, 3)
            # 计算X方向梯度
            gx[n, m] = conv2d(subrect, op_x)
            # 计算Y方向梯度
            gy[n, m] = conv2d(subrect, op_y)
            # 计算变化幅度
            amplitude[n, m] = math.sqrt(gx[n, m] * gx[n, m] + gy[n, m] * gy[n, m])
            # 计算变化角度
            angle[n, m] = solve_angle(gx[n, m], gy[n, m])

    # 非极大信号压制（幅度）
    print("非极大信号压制")
    for n in range(2, image.shape[0] - 1):
        for m in range(2, image.shape[1] - 1):
            if gx[n, m] == 0:
                m0, m1 = get_pixel(amplitude, n - 1, m), get_pixel(amplitude, n + 1, m)
            elif gy[n, m] == 0:
                m0, m1 = get_pixel(amplitude, n, m - 1), get_pixel(amplitude, n, m + 1)
            else:
                g = gy[n, m] / gx[n, m]
                if 0 <= g < 1:
                    m0 = get_pixel(amplitude, n, m + 1)
                    m1 = get_pixel(amplitude, n, m - 1)
                elif 1 <= g:
                    m0 = get_pixel(amplitude, n - 1, m)
                    m1 = get_pixel(amplitude, n + 1, m)
                elif g <= -1:
                    m0 = get_pixel(amplitude, n - 1, m)
                    m1 = get_pixel(amplitude, n + 1, m)
                else:
                    m0 = get_pixel(amplitude, n, m + 1)
                    m1 = get_pixel(amplitude, n, m - 1)
            if amplitude[n, m] < m0 or amplitude[n, m] < m1:
                amplitude[n, m] = 0

    # 基于BFS的边缘连接
    def follow(row, col):
        queue = [(row, col)]
        result[row, col] = 255 if amplitude[row, col] > 255 else amplitude[row, col]
        while len(queue) > 0:
            row, col = queue.pop(0)
            for n in range(-1, 2):
                for m in range(-1, 2):
                    if 0 <= row + n < image.shape[0] and 0 <= col + m < image.shape[1] and \
                            (n != 0 or m != 0) and amplitude[row + n, col + m] >= tl and result[row + n, col + m] == 0:
                        result[row + n, col + m] = int(amplitude[row + n, col + m])
                        result[row + n, col + m] = 255 if result[row + n, col + m] > 255 else result[row + n, col + m]
                        queue.append((row + n, col + m))

    # 双阈值边缘连接
    print("双阈值边缘连接")
    for n in range(image.shape[0]):
        for m in range(image.shape[1]):
            if amplitude[n, m] >= th and result[n, m] == 0:
                follow(n, m)

    print("输出二值图")
    return binaryscale(result).astype(np.uint8)


# import scipy.misc
#
# img = scipy.misc.imread("cat.jpg")
#
# print("模糊")
# img = blur.gauss_blur(img, 5)
#
# print("开始边缘提取")
# img = canny(img, 50, 150)
# img = scipy.misc.imsave("data/canny/cat.jpg", img, "JPEG")
# print("完成")
