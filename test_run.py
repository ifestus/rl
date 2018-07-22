import gym
from skimage import transform
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np

# import argparse

from dqn import DQN

# parser = argparse.ArgumentParser()
# parser.add_argument('--chkp-file', help='saved checkpoint for model to run',
#                     type=str)

# args = parser.parse_args()
env = gym.make('Pong-v0')

sess = tf.Session()

# store 3 previous frames of experience
_D = []
_m = 3
_X_size = (84, 84, 1)
_EXP_REPLAY_FRAMES = 4

checkpoint = '/home/merlin/models/model.ckpt'

DQN_estimate = DQN(sess, name='estimate')
DQN_estimate.load_model(checkpoint)

chkp.print_tensors_in_checkpoint_file(checkpoint, tensor_name='',
                                      all_tensors=True, all_tensor_names=True)


def action_from_dqn(obs):
    X = np.concatenate([_D[x][0] for x in range(_EXP_REPLAY_FRAMES-_m,
                                                _EXP_REPLAY_FRAMES, 1)],
                       -1)
    X = np.expand_dims(np.append(X, obs, -1), 0)
    action_values = DQN_estimate.action_values(X)
    action = np.argmax(action_values, 1)[0]

    return action


def take_action(obs, action):
    next_obs, reward, done, info = env.step(action)
    obs_pre = transform.resize(next_obs, _X_size)

    r = np.clip(reward, -1, 1)

    experience = (obs, action, r, obs_pre, done)
    if len(_D) >= _EXP_REPLAY_FRAMES:
        _D.pop()
    _D.append(experience)
    return obs_pre


for episode in range(1):
    obs_pre = transform.resize(env.reset(), _X_size)
    action = 0
    reward = 0
    done = 0

    for t in range(4):
        action = env.action_space.sample()
        obs_pre = take_action(obs_pre, action)

    while not done:
        env.render()
        action = action_from_dqn(obs_pre)
        obs_pre = take_action(obs_pre, action)

DQN_estimate.close()
