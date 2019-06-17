import os
import numpy as np
from PIL import Image


def name2vec(name):
    ans = ""
    for i in name:
        ans += str(i)
    return ans


def convert2(img, threshold=170):
    """
    将彩色图片变为灰度图片
    :param img:彩色图片 输入为np.array
    :return: 一张灰度图片
    """
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray[gray <= threshold] = 0
        gray[gray > threshold] = 255
        return gray
    else:
        return img


def text2vec(text):
    """
    将文字转化为向量
    :param text: 文字 example： 7895
    :return: 向量 example: [[0 0 0 0 0 0 0 0 0 1]
                           [0 1 0 0 0 0 0 0 0 0]
                           [0 0 0 0 0 1 0 0 0 0]
                           [0 0 1 0 0 0 0 0 0 0]]
    """
    text_len = len(text)
    if text_len > 4:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(40)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * 10 + char2pos(c)
        vector[idx] = 1
    return vector


def listImg(path):
    """
    读出目录下所有的图片
    :param path: 目录
    :return: x 图片数据
             y label数据
    """
    train_x, train_y = [], []
    imgs = os.listdir(path)
    l = len(imgs)
    for img in imgs:
        if os.path.splitext(img)[1] == '.jpg':
            train_x.append(
                convert2(np.array(Image.open(os.path.join(path, img)))))
            train_y.append(text2vec(os.path.splitext(img)[0]))

    return np.reshape(np.array(train_x), (l, 20, 60, 1)), np.array(train_y)
