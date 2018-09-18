# encoding=utf-8
import json, time, requests, os, sys
from io import BytesIO
import numpy as np
from PIL import Image
# from cnn import crack_capture
from test import pre

EXAM_NO = os.getenv('EXAM_NO')  # 准考证号
EXAMINNE_NO = os.getenv('EXAMINNE_NO')  # 考生号
MESSAGE_ADDRESS = 'http://127.0.0.1:5000'  # 微信接收地址
USERNAME = os.getenv('USERNAME')  # 发送人USERNAME

print('hello world')
sys.stdout.flush()


def sendMessage(message):
    """
    通过微信发送信息
    :param message:信息文件
    """
    requests.post(MESSAGE_ADDRESS, {'message': message, 'user': USERNAME})



def getCookie():
    """
    拿到网站的Cookie
    :return: cookie
    """
    return requests.get('http://query.bjeea.cn/queryService/rest/admission/110').cookies['JSESSIONID1']


def get_capture(headers):
    """
        请求网站验证码，并验证
    :param headers: 请求头
    :return: 验证码
    :return  验证码是否正确
    """

    def convert2gray(img):
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    def test(capture):
        r = requests.post('http://query.bjeea.cn/captcha', {'param': capture, 'name': 'capture'}, headers=headers)
        return r.text

    Get_img = requests.get('http://query.bjeea.cn/captcha.jpg', headers=headers)
    img = Image.open(BytesIO(Get_img.content))
    capture = np.reshape(convert2gray(np.array((img))), (1, 20, 60))
    predict = pre(capture)
    return predict, test(predict)


# 每分钟查询一次
if __name__ == '__main__':
    headers = {'Cookie': 'JSESSIONID1=' + getCookie()}
    while True:
        capture, test = '', ''
        while test != '{"status":"y", "info":"&nbsp;"}':
            capture, test = get_capture(headers)

        r = requests.post('http://query.bjeea.cn/queryService/rest/admission/110',
                          {'examNo': EXAM_NO, 'examinneNo': EXAMINNE_NO, 'captcha': capture, 'examId': '4865'},
                          headers=headers
                          )

        result = json.loads(r.text)
        try:
            data = result['enrollList'][0]
            sendMessage('姓名：' + data['NAME'])
            sendMessage('准考证号：' + data['EXAM_NO'])
            sendMessage('考生号：' + data['EXAMINNE_NO'])
            sendMessage('录取批次：' + data['GRADE8'])
            sendMessage('院校代码：' + data['GRADE10'])
            sendMessage('录取院校：' + data['GRADE11'])
            sendMessage('专业代码：' + data['GRADE12'])
            sendMessage('录取专业：' + data['GRADE13'])
            break
        except:
            print('continue .....')
            sys.stdout.flush()
            time.sleep(60)
