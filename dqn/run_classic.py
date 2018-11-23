import gym
# from skimage import transform
import tensorflow as tf

import numpy as np

from dqn import DQN
from helpers import action_from_dqn,
                    take_step,
                    craft_input_with_replay_memory,
                    sample_from_experience,
                    gen_y,
                    update_models_from_experience
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
        observation, reward, done, info = _GYM_ENV.reset()
        action = 0
        t = 0
        while True:
            _ = np.random.sample()
            if _ > _epsilon and not done:
                model_input = craft_input_from_dqns({
                    'obs': observation,
                    'experience_replay': _D,
                    'm': _M,
                    'estimate_dqn': _DQN_ESTIMATE,
                    'metrics': _METRICS
                })
                action = action_from_dqn(model_input, _DQN_ESTIMATE, _METRICS)
            else:
                action = _GYM_ENV.action_space.sample()

            observation, reward, done, info = take_step({
                'action': action,
                'done': done,
                't': t,
                'obs': observation,
                'gym_env': _GYM_ENV,
                'experience_replay': _D,
                'num_replay_max_size': _EXP_REPLAY_FRAMES,
                'transform_size': False,
                'metrics': _METRICS
            })

            # Update epsilon value after random filling of experience pool
            # after frame _EXP_REPLAY_FRAMES*2, epsilon should be == .1
            if t >= _EXP_REPLAY_FRAMES and t < _EXP_REPLAY_FRAMES*2:
                _epsilon -= (.9)/_EXP_REPLAY_FRAMES

            if t >= _REPLAY_START_SIZE:
                update_models_from_experience(t)

            # Incriment our episodic time counter
            _METRICS.inc_episodic_t()
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
