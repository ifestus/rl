import numpy as np
import tensorflow as tf

import logging

class DQN(object):
   def __init__(self, behavior_policy='e-greedy', gamma=0.99, k=4, batch_size=32, recall_frames=1000, clip=True):
      self.behavior_policy = behavior_policy
      self.gamma = gamma
      self.k = k
      self.batch_size = batch_size
      self.recall_frames = recall_frames
      self.clip = clip

   def build_ddn(self, ):
      pass

