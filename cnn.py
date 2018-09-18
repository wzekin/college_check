import os
import random

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.framework import graph_util

y_ = tf.placeholder(tf.float32, shape=[None, 40], name='y_')
keep_prob = tf.placeholder(tf.float32, name="keep")

x_ = tf.placeholder(tf.float32, shape=[None, 20, 60], name='input_x')  # [batch_size, height, width, channels]


def convert2gray(img):
    """
    将彩色图片变为灰度图片
    :param img:彩色图片 输入为np.array
    :return: 一张灰度图片
    """
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
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
            train_x.append(convert2gray(np.array(Image.open(os.path.join(path, img)))))
            train_y.append(text2vec(os.path.splitext(img)[0]))

    return np.reshape(np.array(train_x), (l, 20, 60)), np.array(train_y)


def capture_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(x_, [-1, 20, 60, 1])

    w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 64]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 256]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([256]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.avg_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_d1 = tf.Variable(w_alpha * tf.random_normal([3 * 8 * 256, 256]))
    b_d1 = tf.Variable(b_alpha * tf.random_normal([256]))
    dense = tf.reshape(conv3, [-1, w_d1.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d1), b_d1))
    dense = tf.nn.dropout(dense, keep_prob)

    w_d2 = tf.Variable(w_alpha * tf.random_normal([256, 128]))
    b_d2 = tf.Variable(b_alpha * tf.random_normal([128]))
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d2), b_d2))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([128, 40]))
    b_out = tf.Variable(b_alpha * tf.random_normal([40]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def train_cnn():
    train_x, train_y = listImg('./img')
    val_x, val_y = listImg('./val')

    def get_next_batch(batch_size=64):
        batch_x = np.zeros([batch_size, 20, 60])
        batch_y = np.zeros([batch_size, 40])
        for i in range(batch_size):
            a = random.randint(0, len(train_x) - 1)
            batch_x[i], batch_y[i] = train_x[a], train_y[a]
        return batch_x, batch_y

    y = capture_cnn()
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
    predict = tf.reshape(y, [-1, 4, 10])
    max_idx_p = tf.argmax(predict, 2, name="pre")
    max_idx_l = tf.argmax(tf.reshape(y_, [-1, 4, 10]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.exists("checkpoint"):
            saver.restore(sess, tf.train.latest_checkpoint("."))
        else:
            sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(20)
            _, loss_ = sess.run([train_op, cost], feed_dict={x_: batch_x, y_: batch_y, keep_prob: 0.5})

            if step % 100 == 0:
                print(step, loss_)
                acc_ = sess.run(acc, feed_dict={x_: val_x, y_: val_y, keep_prob: 1})
                print('acc:', acc_)
                if acc_ > 0.90:
                    saver.save(sess, "./model.ckpt", global_step=step)
                    break
                if step % 1000 == 0:
                    saver.save(sess, "./model.ckpt", global_step=step)
            step += 1
        save_pb(sess)


def save_pb(sess):
    output_graph = "model.pb"
    op = sess.graph.get_operations()
    print(op[0].name)
    output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess,
        sess.graph.as_graph_def(),
        ["input_x", "pre", "keep"]  # 需要保存节点的名字
    )
    with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出


if __name__ == '__main__':
    train_cnn()
else:
    output = capture_cnn()
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint("."))
