import os
import time
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from cnn import predict as pre
from main import getCookie

Get_path = './img/'
Get_number = 400
Get_url = 'http://query.bjeea.cn/captcha.jpg'


def GetImgCode():
    """
    下载验证码 手动标记
    """

    def convert2gray(img):
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    def test(capture):
        r = requests.post('http://query.bjeea.cn/captcha', {
            'param': capture,
            'name': 'capture'
        },
                          headers=headers,
                          timeout=1)
        return r.text

    headers = {'Cookie': 'JSESSIONID1=' + getCookie()}
    get_img_start = time.time()
    if os.path.isdir(Get_path):
        pass
    else:
        os.makedirs(Get_path)
        print('下载目录不存在，创建目录中----------')
        print('下载目录创建成功，目录名->' + Get_path)
    if Get_url != '':
        print("获取下载链接成功---------")
        print("开始下载验证码")
        for i in range(0, Get_number):
            time.sleep(1)
            try:
                print("下载第" + str(i) + "张验证码")
                Get_img = requests.get(Get_url, headers=headers, timeout=1)
                img = Image.open(BytesIO(Get_img.content))
                capture = np.reshape(convert2gray(np.array((img))),
                                     (1, 20, 60))
                predict = pre(capture)
                print(predict)
                if test(predict) != '{"status":"y", "info":"&nbsp;"}':
                    continue
                filePath = Get_path + predict + '.jpg'
                with open(filePath, 'bw') as f:
                    f.write(Get_img.content)
            except Exception as e:
                print(e)
        get_img_end = time.time()
        print("已完成，共下载" + str(Get_number) + "张验证码---------")
        print("执行时间 %f m" % (get_img_end - get_img_start))

    else:
        print('验证码下载地址为空')
        exit()


GetImgCode()
