import numpy as np
import baselines

import argparse
import logging

class Sarsa(object):
    def __init__(self, action_space, policy='e-greedy', epsilon=0.01, lr=0.01):
        self.action_space = action_space
        self.epsilon = epsilon
        self.learning_rate = lr
        self.policy = policy # only e-greedy currently

    def act(self, observation, reward, done):
        pass

    def step(self, ):
        pass

class QLearning(object):
    def __init__(self, action_space, policy='e-greedy', epsilon=0.01):
        self.action_space = action_space
        self.epsilon = epsilon
        self.policy = policy # only e-greedy currently

    def act(self, observation, reward, done):
        pass

    def step(self):
        pass

class TRPOAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        pass

class PPOAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        pass

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
