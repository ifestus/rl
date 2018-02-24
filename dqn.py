from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

export_dir = ('/home/merlin/rl/models')

    #def __init__(self, behavior_policy='e-greedy', epsilon = 0.05, gamma=0.99,
    #             m=4, batch_size=32, k=4, valid_actions=5, recall_frames=1000,
    #             clip=True, C=50):
class DQN(object):
    def __init__(self, lr=.00025, gamma=0.99, m=4, batch_size=32, valid_actions=6, clip=True, name):
        self.learning_rate = .00025
        self.gamma = gamma
        self.m = m
        self.batch_size = batch_size
        self.valid_actions = valid_actions
        self.clip = clip
        self.name = name

        self.builder = tf.saved_model_builder.SavedModelBuilder(export_dir)

    def build_dqn(self):
        with tf.name_scope(name):
            input_layer = tf.placeholder([batch_size, 84, 84, m], tf.float32)

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
            out = tf.layers.dense(inputs=dense1,
                                  units=self.valid_actions)
            self.out = out
            self.predict = tf.argmax(out, 1)
            self.Y = tf.placeholder(tf.float32, [None, valid_actions])
            self.loss = tf.mean_squared_error(Y, self.predict)
            self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss)

        # with tf.Session(graph=self.graph) as sess:
        #     sess.run(tf.global_variables_initializer())
        #     self.builder.add_meta_graph_and_variables(sess, ["estimate"])

        self.builder.save()
