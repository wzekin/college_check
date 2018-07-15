import tensorlayer as tl
import tensorflow as tf


#  cnn模型
def model():
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 20, 60, 1], name='x')  # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.float32, shape=[None, 40], name='y_')
    net = tl.layers.InputLayer(x, name='input')
    ## Simplified conv API (the same with the above layers)
    net = tl.layers.Conv2d(net, 32, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn1')
    net = tl.layers.MeanPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    net = tl.layers.Conv2d(net, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn3')
    net = tl.layers.MeanPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool3')
    ## end of conv
    net = tl.layers.FlattenLayer(net, name='flatten')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
    net = tl.layers.DenseLayer(net, 256, act=tf.nn.relu, name='relu2')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop3')
    net = tl.layers.DenseLayer(net, 40, act=None, name='output')
    return sess, net, x, y_
