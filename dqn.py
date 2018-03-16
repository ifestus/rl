# architecture idea from https://github.com/DongjunLee/dqn-tensorflow/
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dqn_models import CNN

import numpy as np
import tensorflow as tf

class DDQN(DQN):
    def __init__(self, session, name="CNN", lr=.00025, gamma=0.99, m=4, batch_size=32, valid_actions=6, clip=True):
        super().__init__(session, name, lr, gamma, m, batch_size, valid_actions, clip)

    def predict(self, X):
        return self.session.run(self.model.predict,
                                feed_dict={self._X: X})

    def update(self, X, Y):
        self.session.run([self._optimizer, self._loss],
                         feed_dict={self._X: X, self._Y: Y})

class DQN(object):
    def __init__(self, session, name="CNN", lr=.00025, gamma=0.99, m=4, batch_size=32, valid_actions=6, clip=True):
        self._X = tf.placeholder(tf.float32, [None, 84, 84, m])

        self.name = name
        self.session = session

        self._gamma = gamma
        self._learning_rate = lr

        self._m = m
        self._batch_size = batch_size
        self._valid_actions = valid_actions

        self._build_model()
        self.session.run(self.model.initializer)

        self._saver = tf.train.Saver(self.get_variables_dict())

    def _build_model(self):
        self.model = CNN(self._X,
                         name=self.name,
                         lr=self._learning_rate,
                         valid_actions=self._valid_actions)
        self.model.build_dqn()

        self._Y = self.model.Y
        self._out = self.model.out
        self._predict = self.model.predict

        self._loss = self.model.loss
        self._optimizer = self.model.optimizer

    def get_variables_dict(self):
        variables = {}
        for v in tf.trainable_variables():
            if self.name in v.name:
                variables[v.name.replace(self.name+'/',"")] = v

        # print("{} variables: \n{}".format(self.name, variables))
        return variables

    def action_values(self, X):
        return self.session.run(self._out,
                                feed_dict={self._X: X})

    def predict(self, X):
        return self.session.run(self.model.predict,
                                feed_dict={self._X: X})

    def update(self, X, Y):
        self.session.run([self._optimizer, self._loss],
                         feed_dict={self._X: X, self._Y: Y})

    def close(self):
        tf.reset_default_graph()
        self.session.close()

    def save_model(self, checkpoint):
        self._saver.save(self.session, checkpoint)

    def load_model(self, checkpoint):
        self._saver.restore(self.session, checkpoint)
