import gym
# from skimage import transform
import tensorflow as tf

import numpy as np

from dqn import DQN
from helpers import action_from_dqn, take_step, craft_input_with_replay_memory, update_models_from_experience, append_to_experience_replay
from metrics import Metrics

# DQN hyper-parameters
_TRAIN_FRAMES = 50000
_EXP_REPLAY_FRAMES = 5000
_REPLAY_START_SIZE = 5000
_GAMMA = 0.99
_MINIBATCH = 32
_D = []  # Experience replay pool
_C = 4000  # time steps between updating target model to estimate model vars
_M = 4  # Agent history length - the number of frames fed into the DQN model
_epsilon = 1.0

# DQN Model and TF Globals
_CHECKPOINT_FILE = '/home/merlin/models/classic_dqn.ckpt'
_TF_SESS = tf.Session()
# _DQN_ESTIMATE = DQN(_TF_SESS, input_shape=[None, 4, _M], name='estimate')
# _DQN_TARGET = DQN(_TF_SESS, input_shape=[None, 4, _M], name='target')
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
    for _ in range(20):
        observation = _GYM_ENV.reset()
        action = 0
        done = 0
        global_t = 0
        while True:
            sample = np.random.sample()
            if sample > _epsilon and not done:
                model_input = craft_input_with_replay_memory({
                    'obs': observation,
                    'experience_replay': _D,
                    'm': _M,
                    'estimate_dqn': _DQN_ESTIMATE,
                    'metrics': _METRICS
                })
                action = action_from_dqn(model_input, _DQN_ESTIMATE, _METRICS)
            else:
                action = _GYM_ENV.action_space.sample()

            prev_obs = observation
            observation, reward, done, info = take_step({
                'action': action,
                'done': done,
                't': global_t,
                'gym_env': _GYM_ENV,
                'transform_size': False,
                'metrics': _METRICS
            })

            append_to_experience_replay({
                'prev_obs': prev_obs,
                'action': action,
                'reward': reward,
                'next_obs_transformed': observation,
                'done': done,
                'experience_replay': _D,
                'experience_replay_max_frames': _EXP_REPLAY_FRAMES
            })

            # Update epsilon value after random filling of experience pool
            # after frame _EXP_REPLAY_FRAMES*2, epsilon should be == .1
            if global_t >= _EXP_REPLAY_FRAMES and global_t < _EXP_REPLAY_FRAMES*2:
                _epsilon -= (.9)/_EXP_REPLAY_FRAMES

            if global_t >= _REPLAY_START_SIZE:
                update_models_from_experience({
                    't': global_t,
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
            global_t += 1
            if done:
                print("episode ended - global_t: {}".format(global_t))
                break


if __name__ == '__main__':
    # file_writer = tf.summary.FileWriter('./tf_graph', _TF_SESS.graph)
    try:
        main()
    finally:
        _DQN_ESTIMATE.close()
        _DQN_TARGET.close()
