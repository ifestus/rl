import gym
# from skimage import transform
import tensorflow as tf

import numpy as np

from dqn import DQN
from metrics import Metrics

# DQN hyper-parameters
_TRAIN_FRAMES = 100000
_EXP_REPLAY_FRAMES = 25000
_GAMMA = 0.99
_MINIBATCH = 32
_D = []  # Experience replay pool
_C = 4000  # time steps between updating target model to estimate model vars
_M = 4  # Number of frames used as input to a model
_epsilon = 1.0

# DQN Model and TF Globals
# _CHECKPOINT_FILE = '/home/merlin/models/classic_dqn.ckpt'
_TF_SESS = tf.Session()
_DQN_ESTIMATE = DQN(_TF_SESS, name='estimate')
_DQN_TARGET = DQN(_TF_SESS, name='target')

# Metrics object
_METRICS = Metrics()

# Create OpenAI Gym environment
_GYM_ENV = gym.make('CartPole-v0')
print(_GYM_ENV.action_space)
print(_GYM_ENV.observation_space)


def main():
    global _epsilon
    for i_episode in range(20):
        observation = _GYM_ENV.reset()
        action = 0
        t = 0
        while True:
            _ = np.random.sample()
            if _ > epsilon:
                action = action_from_dqn(obs)
            else:
                action = _GYM_ENV.action_space.sample()

            obs = take_step(obs, t, action)

            # Update epsilon value after random filling of experience pool
            # after frame _EXP_REPLAY_FRAMES*2, epsilon should be == .1
            print(observation)
            action = _GYM_ENV.action_space.sample()
            observation, reward, done, info = _GYM_ENV.step(action)
            t += 1
            if done:
                print('Episode finished after {} timesteps'.format(t+1))
                break


if __name__ == '__main__':
    # file_writer = tf.summary.FileWriter('./tf_graph', _TF_SESS.graph)
    try:
        main()
    except Exception as err:
        print('Ran into an issue: {}'.format(err))
        print('Attempting to clean up')
        _DQN_ESTIMATE.close()
        _DQN_TARGET.close()
