import gym
from skimage import transform
import tensorflow as tf
# from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np

import argparse

from dqn import DQN
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


def action_from_dqn(obs):
    # _EXP_REPLAY_FRAMES-_M+1 because we're going to grab the most recent
    # _M-1 frames from the top of _D and append our observation to that
    X = np.concatenate([_D[x][0] for x in range(_EXP_REPLAY_FRAMES-_M+1,
                                                _EXP_REPLAY_FRAMES, 1)],
                       -1)
    X = np.expand_dims(np.append(X, obs, -1), 0)
    action_values = _DQN_ESTIMATE.action_values(X)
    action = np.argmax(action_values, 1)[0]

    _METRICS.add_to_values_accum(np.reshape(action_values, (6))[action])

    return action


def take_step(obs, t, action):
    next_obs, reward, done, info = _GYM_ENV.step(action)
    if not done:
        next_obs_transformed = transform.resize(next_obs, _TRANSFORM_SIZE)

        r = np.clip(reward, -1, 1)
        _METRICS.add_to_rewards_accum(r)

        experience = (obs, action, r, next_obs_transformed, done)
        if len(_D) >= _EXP_REPLAY_FRAMES:
            _D.pop(0)
        _D.append(experience)

    else:
        _t = _METRICS.get_episodic_t()
        print('Episode finished after {} timesteps ({}).'.format(t+1, _t))

        next_obs_transformed = transform.resize(_GYM_ENV.reset(),
                                                _TRANSFORM_SIZE)

        _METRICS.write_metrics()
        _METRICS.set_rewards_accum(0.0)
        _METRICS.set_values_accum(0.0)
        _METRICS.set_episodic_t(0)

    return next_obs_transformed


def sample_from_experience():
    sample = {'st': [], 'at': [], 'rt+1': [], 'st+1': [], 'dt+1': []}

    # We want to add ${_MINIBATCH} experience tuples to our sample
    for _ in range(_MINIBATCH):
        # k is th eindex into our experience pool _D
        # we don't want to start our sample with a done frame because that
        # makes things a bit more complicated.
        k = np.maximum(np.randint(_EXP_REPLAY_FRAMES), _M-1)

        # If any of the frames from k-m+1 to k-1 (inclusive) are done frames
        if np.amax(_D[k-_M+1:k][1][4]) is True:
            done_frame = np.argmax(_D[k-_M+1:k][1][4])
            new_k = k - done_frame

            # shift k such that the done frame is new k
            if k - done_frame > _M-1:
                k = new_k
            else:
                k = np.maximum(np.random.randint(_EXP_REPLAY_FRAMES), _M-1)

        st = []
        at = []
        rt1 = []
        st1 = []
        dt1 = 0

        for x in range(k, k-_M, -1):
            # st and st1 will each have to be np.concatenated to create a
            # (84, 84, _M)-shape np array
            st.insert(0, _D[x][0])
            st1.insert(0, _D[x][3])

            at.insert(0, _D[x][1])
            rt1.insert(0, _D[x][2])

            dt1 = dt1 or _D[x][4]

        sample['st'].append(np.concatenate(st, -1))
        sample['at'].append(at)
        sample['rt+1'].append(rt1)
        sample['st+1'].append(np.concatenate(st1, -1))
        sample['dt+1'].append(dt1)

    # Lets turn those pesky lists in sample into np arrays
    for key in sample:
        sample[key] = np.array(sample[key])

    return sample


def gen_y(sample):
    rt1 = sample['rt+1'][:][-1]
    done = sample['dt+1']

    # Input to the target DQN is t+1 state
    X = sample['st+1']

    Y = rt1 + _GAMMA * np.multiply(done,
                                   np.amax(_DQN_TARGET.action_values(X), -1))

    return np.reshape(Y, (_MINIBATCH, 1))


def update_models_from_experience(t):
    # Update models after we fill experience replay and only on every 4th frame
    if t >= _EXP_REPLAY_FRAMES and t+1 % 4 == 0:
        sample = sample_from_experience()
        X = sample['st']
        Y = gen_y(sample)

        if (t+1) % 12000 == 0:
            print('Y for sampled values: ', Y)

        # Update estimate model
        _DQN_ESTIMATE.update(X, Y)

        # Reset Q_TARGET = Q_ESTIMATE
        if t+1 % _C == 0:
            _DQN_ESTIMATE.save_model(_CHECKPOINT_FILE)
            _DQN_TARGET.load_model(_CHECKPOINT_FILE)


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

        # for t in range(_TRAIN_FRAMES + 2*_EXP_REPLAY_FRAMES):
        while True:
            _ = np.random.sample()
            if _ > _epsilon:
                action = action_from_dqn(obs)
            else:
                action = _GYM_ENV.action_space.sample()

            obs = take_step(obs, t, action)

            # Update epsilon value after random filling of experience pool
            # after frame _EXP_REPLAY_FRAMES*2, epsilon should be == .1
            if t >= _EXP_REPLAY_FRAMES and t < _EXP_REPLAY_FRAMES*2:
                _epsilon -= (.9)/_EXP_REPLAY_FRAMES

            update_models_from_experience(t)

            # Incriment our episodic time counter
            _METRICS.inc_episodic_t()
            t += 1


if __name__ == '__main__':
    file_writer = tf.summary.FileWriter('./tf_graph', _TF_SESS.graph)
    main()

    _DQN_ESTIMATE.close()
    _DQN_TARGET.close()
