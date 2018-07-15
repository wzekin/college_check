import os

import tensorflow as tf
import numpy as np
from PIL import Image
import tensorlayer as tl

from model import model

sess, net, x, y_ = model()

def convert2gray(img):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text):
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
    train_x, train_y = [], []
    imgs = os.listdir(path)
    l = len(imgs)
    for img in imgs:
        if os.path.splitext(img)[1] == '.jpg':
            train_x.append(convert2gray(np.array(Image.open(os.path.join(path, img)))))
            train_y.append(text2vec(os.path.splitext(img)[0]))

    return np.reshape(np.array(train_x), (l, 20, 60, 1)), np.array(train_y)




if __name__ == '__main__':
    train_x, train_y = listImg('./img')
    val_x, val_y = listImg('./val')


    y = net.outputs

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
    predict = tf.reshape(y, [-1, 4, 10])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y_, [-1, 4, 10]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    train_params = net.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

    # initialize all variables in the session
    tl.layers.initialize_global_variables(sess)

    # print network information
    net.print_params()
    net.print_layers()
    tl.utils.fit(sess, net, train_op, cost, train_x, train_y, x, y_, X_val=val_x, y_val=val_y, acc=acc, batch_size=20,
                 n_epoch=500, print_freq=5)
    tl.files.save_npz(net.all_params, name='model1.npz')
    sess.close()