import random

import numpy as np
from utils import *

import tensorflow as tf


def capture_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,
                               5,
                               activation="relu",
                               input_shape=(20, 60, 1)),
        tf.keras.layers.Conv2D(20, 1, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.Conv2D(20, 1, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(40, activation='relu'),
    ])


def my_acc(y_true, y_pred):
    return tf.equal(tf.argmax(tf.reshape(y_true, [-1, 4, 10]), 2),
                    tf.argmax(tf.reshape(y_pred, [-1, 4, 10]), 2))


def train_cnn():
    train_x, train_y = listImg('./img')
    val_x, val_y = listImg('./val')

    model = capture_cnn()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=[my_acc])
    model.summary()
    model.fit(train_x,
              train_y,
              epochs=100,
              batch_size=32,
              validation_data=(val_x, val_y))
    model.save('model.h5')


def predict(image):
    image = np.reshape(image, (1, 20, 60, 1))
    result = model.predict(np.reshape(image, (1, 20, 60, 1)))
    result = np.argmax(np.reshape(result, (4, 10)), 1)
    return name2vec(result)


if __name__ == '__main__':
    train_cnn()
else:
    model = tf.keras.models.load_model('model.h5',
                                       custom_objects={'my_acc': my_acc})
