from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import logging

class DQN(object):
    def __init__(self, behavior_policy='e-greedy', epsilon = 0.05, gamma=0.99, m=4, batch_size=32, k=4, valid_actions=5, recall_frames=1000, clip=True):
        self.behavior_policy = behavior_policy
        self.gamma = gamma
        self.k = k
        self.batch_size = batch_size
        self.recall_frames = recall_frames
        self.clip = clip

    def build_ddn(self):
        input_layer = tf.placeholder([batch_size, 84, 84, m], tf.int32)
        # First convolutional layer -- Input layer
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=32,
                                 strides=[1, 4, 4, 1],
                                 kernel_size=[8, 8],
                                 padding="valid", # try same, too
                                 activation=tf.nn.relu,
                                 name="input_conv")

        # Second convolutional layer. Input is the output of the previous layer.
        conv2 = tf.layers.conv2d(inputs=conv1,
                                 filters=64,
                                 strides=[1, 2, 2, 1],
                                 kernel_size=[4, 4],
                                 padding="valid", # try same, too
                                 activation=tf.nn.relu,
                                 name="conv2")

        # Third convolutional layer
        conv3 = tf.layers.conv2d(inputs=conv2,
                                 filters=64,
                                 strides=np.ones([4]),
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
        output_layer = tf.layers.dense(inputs=dense1,
                                       units=self.valid_actions)

        values = output_layer


        return self.model

