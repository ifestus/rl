import gym
from skimage import transform
import tensorflow as tf
# from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np

import argparse

from dqn import DQN
from helpers import gen_y, sample_from_experience, take_step, action_from_dqn, craft_input_with_replay_memory
from metrics import Metrics

# DQN hyper-parameters
_TRAIN_FRAMES = 250000
_EXP_REPLAY_FRAMES = 125000
_GAMMA = 0.99
_MINIBATCH = 32
_D = []  # Experience Pool
_C = 4000  # time steps between updating target model to estimate model vars
_TRANSFORM_SIZE = (84, 84, 1)  # Transform observations into this size
_M = 4  # Number of frames used as input to a model
_epsilon = 1.0

# DQN Model and TF globals
_CHECKPOINT_FILE = '/home/merlin/models/model.ckpt'
_TF_SESS = tf.Session()
_DQN_ESTIMATE = DQN(_TF_SESS, name='estimate')
_DQN_TARGET = DQN(_TF_SESS, name='target')

# Metrics object
_METRICS = Metrics()

# create OpenAI Gym environment
_GYM_ENV = gym.make('Pong-v0')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',
                        help='load variables from previously saved checkpoint',
                        action='store_true')
    args = parser.parse_args()
    return args


def main():
    global _epsilon
    config = get_args()
    if config.load:
        _DQN_ESTIMATE.load_model(_CHECKPOINT_FILE)
        _DQN_TARGET.load_model(_CHECKPOINT_FILE)
    else:
        _DQN_ESTIMATE.save_model(_CHECKPOINT_FILE)
        _DQN_TARGET.load_model(_CHECKPOINT_FILE)

    for episode in range(1000):
        obs = transform.resize(_GYM_ENV.reset(), _TRANSFORM_SIZE)
        action = 0
        t = 0

        while True:
            _ = np.random.sample()
            if _ > _epsilon:
                model_input = craft_input_with_replay_memory({
                    'obs': obs,
                    'experience_replay': _D,
                    'm': _M,
                    'estimate_dqn': _DQN_ESTIMATE,
                    'metrics': _METRICS
                })
                action = action_from_dqn({
                    'input_object': model_input,
                    'estimate_dqn': _DQN_ESTIMATE,
                    'metrics': _METRICS
                })
            else:
                action = _GYM_ENV.action_space.sample()

            obs, r, done, info = take_step({
                'action': action,
                't': t,
                'obs': obs,
                'done': done,
                'gym_env': _GYM_ENV,
                'experience_replay': _D,
                'experience_replay_max_frames': _EXP_REPLAY_FRAMES,
                'transform_size': _TRANSFORM_SIZE,
                'metrics': _METRICS
            })

            # Update epsilon value after random filling of experience pool
            # after frame _EXP_REPLAY_FRAMES*2, epsilon should be == .1
            if t >= _EXP_REPLAY_FRAMES and t < _EXP_REPLAY_FRAMES*2:
                _epsilon -= (.9)/_EXP_REPLAY_FRAMES

            # if t >= _REPLAY_START_SIZE:
            update_models_from_experience({
                't': t,
                'experience_replay_max_frames': _EXP_REPLAY_FRAMES,
                'estimate_dqn': _DQN_ESTIMATE,
                'target_dqn': _DQN_TARGET,
                'checkpoint_file': _CHECKPOINT_FILE,
                'update_freq': _C,
                'minibatch_size': _MINIBATCH,
                'm': _M
            })

            # Incriment our episodic time counter
            _METRICS.inc_episodic_t()
            t += 1


if __name__ == '__main__':
    file_writer = tf.summary.FileWriter('./tf_graph', _TF_SESS.graph)
    main()

    _DQN_ESTIMATE.close()
    _DQN_TARGET.close()
