# architecture idea from https://github.com/DongjunLee/dqn-tensorflow/
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dqn_model import CNN

import numpy as np
import tensorflow as tf

export_dir = ('/home/merlin/rl/models')

class DQN(object):
    def __init__(self, name, session=tf.Session(), lr=.00025, gamma=0.99, m=4, batch_size=32, valid_actions=6, clip=True):
        self.X = tf.paceholder([-1, 84, 84, m])

        self.name = name
        self.session = session

        self.gamma = gamma
        self.learning_rate = lr

        self.m = m
        self.batch_size = batch_size
        self.valid_actions = valid_actions

        self.build_model()

    def _build_model(self):
        self.model = CNN(learning_rate=self.learning_rate,
                         valid_actions=self.valid_actions)

    def predict(self):
        pass

    def update(self):
        pass

    def close(self):
        tf.reset_default_graph()
        self.session.close()

    def save_model(self):
        self.builder.add_meta_graph_and_variables(self.session, [self.name])
        self.builder.save()

    def load_model(self):
        pass
