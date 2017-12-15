# -*- coding:utf-8 -*-

from PIL import Image, ImageFont, ImageDraw, ImageFilter
import random, math
import numpy as np


# 返回随机字母
def charRandom():
    return chr((random.randint(65, 90)))


# 返回随机数字
def numRandom():
    return random.randint(0, 9)


# 随机颜色
def colorRandom1():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))


# 随机长生颜色2
def colorRandom2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))


width = 40 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))
# 创建font对象
font = ImageFont.truetype('Arial.ttf', 36)

# 创建draw对象
draw = ImageDraw.Draw(image)
# 填充每一个颜色
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=colorRandom1())

# 输出文字
for t in range(4):
    draw.text((10 + t * 40, 10), charRandom(), font=font, fill=colorRandom2())

image.save('./asserts/code.jpg', 'jpeg')

import cv2
import edge, morphology

gray_image = cv2.imread("asserts/code.jpg", 0)

gray_image = cv2.blur(gray_image, (5, 5))
# img = cv2.bilateralFilter(img, 5, 75, 100)     # 参数为：滤波直径（奇数），空间距离标准差，像素距离标准差

gray_image = edge.binaryscale(gray_image)

cv2.imwrite('./asserts/bin.jpg', gray_image)


def kmeans_split(img: np.ndarray, mean_num):
    means_y = np.array([random.randint(0, img.shape[0]) for _ in range(mean_num)])
    means_x = np.array([random.randint(0, img.shape[1]) for _ in range(mean_num)])

    # K-Mean算法每次迭代步骤
    def mean_sum(means_x, means_y):
        mean_set = [[] for i in range(mean_num)]  # 各个分类集
        sum_dist = 0  # 所有节点离自己的mean节点的距离总和
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if img[row, col] == 0:
                    # 每个节点分类为与其距离最近的mean节点所属的分类
                    _min, _idx = None, None
                    for i in range(mean_num):
                        # 寻找最近的mean节点
                        dist = (row - means_y[i]) * (row - means_y[i]) + (col - means_x[i]) * (col - means_x[i])
                        if _min is None or _min > dist:
                            _min = dist
                            _idx = i
                    mean_set[_idx].append((row, col))
                    sum_dist += _min
        return sum_dist, mean_set

    min_dist_sum = None
    shift_sum = 100000
    while shift_sum != 0:
        dist_sum, mean_set = mean_sum(means_x, means_y)
        if min_dist_sum == None or min_dist_sum > dist_sum:
            min_dist_sum = dist_sum
        shift_sum = 0
        # Shift操作
        for i in range(mean_num):
            sum_x, sum_y = 0, 0
            for each_node in mean_set[i]:
                sum_y += each_node[0]
                sum_x += each_node[1]
            pre_x = means_x[i]
            pre_y = means_y[i]
            if len(mean_set[i]) == 0:
                means_x[i] = random.randint(0, img.shape[1])
                means_y[i] = random.randint(0, img.shape[0])
            else:
                means_x[i] = sum_x / len(mean_set[i])
                means_y[i] = sum_y / len(mean_set[i])
            shift_sum += means_y[i] - pre_y + means_x[i] - pre_x

    return sorted([(means_x[i], means_y[i]) for i in range(mean_num)], key=lambda point : point[0]), min_dist_sum


mean_num = 4
_min = None
_mean_points = None
# 做多次kmeans
for i in range(50):
    if i % 10 == 0:
        print("第 %d 次 K-Mean..." % (i+1))
    mean_points, dist_sum = kmeans_split(gray_image, mean_num)
    if _min is None or _min > dist_sum:
        _mean_points = mean_points
        _min = dist_sum


img = Image.open("asserts/bin.jpg")
x = 0
y = 0
for i in range(mean_num):
    if i == 0:
        x = 2 * _mean_points[i][0] - (_mean_points[i+1][0] + _mean_points[i][0]) / 2
        x = max(x, 0)
    if i == mean_num - 1:
        next_x = 2 * _mean_points[i][0] - (_mean_points[i][0] + _mean_points[i - 1][0]) / 2
        next_x = min(next_x, width)
    else:
        next_x = (_mean_points[i + 1][0] + _mean_points[i][0]) / 2
    region = (x, y, next_x, height)
    cropimg = img.crop(region)
    cropimg.save('asserts/%d.jpg' % i)
    x = next_x