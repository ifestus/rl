from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class CNN(object):
    def __init__(self, X, name="CNN", lr=.00025, valid_actions=6):
        self._X = X
        self.name = name
        self._learning_rate = lr
        self._valid_actions = valid_actions

    def build_dqn(self):
        with tf.variable_scope(self.name):
            # input_layer = tf.placeholder([batch_size, 84, 84, m], tf.float32)
            input_layer = self._X

            # First convolutional layer -- Input layer
            conv1 = tf.layers.conv2d(inputs=input_layer,
                                     filters=32,
                                     strides=[4, 4],
                                     kernel_size=[8, 8],
                                     padding="valid", # try same, too
                                     activation=tf.nn.relu,
                                     name="input_conv")

            # Second convolutional layer. Input is the output of the previous layer.
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=64,
                                     strides=[2, 2],
                                     kernel_size=[4, 4],
                                     padding="valid", # try same, too
                                     activation=tf.nn.relu,
                                     name="conv2")

            # Third convolutional layer
            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=64,
                                     strides=np.ones([2]),
                                     kernel_size=[3, 3],
                                     padding="valid", # try same, too
                                     activation=tf.nn.relu,
                                     name="conv3")

            # Dense hidden layer with 512 rectifier units
            conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * 64])
            dense1 = tf.layers.dense(inputs=conv3_flat,
                                     units=512,
                                     activation=tf.nn.relu)

            # Output layer
            dense2 = tf.layers.dense(inputs=dense1,
                                     units=self._valid_actions,
                                     activation=tf.nn.relu)

            self.out = dense2
            self.predict = tf.argmax(self.out, 1)

            self.Y = tf.placeholder(tf.float32, self.out.shape)

            self.loss = tf.losses.mean_squared_error(self.Y, self.out)

            self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=self._learning_rate).minimize(self.loss)

            self.initializer = tf.global_variables_initializer()
