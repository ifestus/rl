import gym
from skimage import transform
import tensorflow as tf
import numpy as np

import argparse
import os

from dqn import DQN

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="load starting point from previous checkpoint",
                    action="store_true")
args = parser.parse_args()

env = gym.make('Pong-v0')

sess = tf.Session()

_TRAIN_FRAMES = 500000
_EXP_REPLAY_FRAMES = 50000
_GAMMA = 0.99
_MINIBATCH = 32

# Metrics will hold tuples. Currently only holds 2-tuples of average rewards
# and action_values over an episode
metrics = []

D = []
M = 50
X_size = (84, 84, 1)
# Number of previous frames we'll take to create a sample for input to DQN
m = 3
epsilon = 1.0

checkpoint = '/home/merlin/models/model.ckpt'

DQN_estimate = DQN(sess, name="estimate")
DQN_target   = DQN(sess, name="target")

if args.load:
    DQN_estimate.load_model(checkpoint)
    DQN_target.load_model(checkpoint)
else:
    DQN_estimate.save_model(checkpoint)
    DQN_target.load_model(checkpoint)

# Prefill D
# D = [(np.zeros((84, 84, 1)), 0, 0, np.zeros((84, 84, 1)), 0)] * _EXP_REPLAY_FRAMES

def sample_experience():
    # Fill the sample with zeros to begin with
    sample = []
    # sample = np.zeros((minibatch_size, X_size[0], X_size[1], X_size[2]))

    # We want to add ${minibatch} experience values to our samples
    for _ in range(_MINIBATCH):
        # k is the index to our experience pool (D)
        k = np.maximum(np.random.randint(_EXP_REPLAY_FRAMES), m)

        # Pull processed frames from the previous ${m-1} experiences
        # and add them to the sample =)
        st = np.expand_dims(
                np.concatenate([D[x][0] for x in range(k-m, k+1, 1)], -1),
                0)
        at = D[k][1]
        rt = D[k][2]
        st1 = np.expand_dims(
                np.concatenate([D[x][3] for x in range(k-m, k+1, 1)], -1),
                0)
        done = D[k][4]
        sample.append((st, at, rt, st1, done))

    return np.array(sample)

def gen_y(sample):
    r = sample[:, 2]
    done = sample[:, 4]

    # Input to the target DQN is t+1 state
    X = np.concatenate(sample[:, 3], 0)

    Y = r + _GAMMA * np.multiply(done, np.amax(DQN_target.action_values(X), -1))

    return np.reshape(Y, (_MINIBATCH, 1))

# Really, we're training based on frames, so we should control the flow as such
for episode in range(1):
    obs_pre = transform.resize(env.reset(), X_size)
    action = 0
    reward = 0

    # Metrics
    _t = 0
    reward_accum = 0.0
    values_accum = 0.0

    for t in range(_TRAIN_FRAMES + 2*_EXP_REPLAY_FRAMES):
        if (t+1) % 5000 == 0:
            print("Time Step: [{}]".format(t+1))
            print("D length", len(D))
        if t >= _TRAIN_FRAMES - _EXP_REPLAY_FRAMES:
            env.render()

        # Get action from the estimate network
        # this action is going to be an e-greedy one that is determined by the
        # estimate network
        obs = obs_pre

        # e-greedy action selection
        _ = np.random.sample()
        if _ > epsilon:
            X = np.concatenate([D[x][0] for x in range(_EXP_REPLAY_FRAMES-m,
                                                       _EXP_REPLAY_FRAMES, 1)],
                               -1)
            X = np.expand_dims(np.append(X, obs, -1), 0)
            action_values = DQN_estimate.action_values(X)
            action = np.argmax(action_values, 1)[0]
            values_accum += action_values[action]
        else:
            action = env.action_space.sample()

        # take action and preprocess next state
        next_obs, reward, done, info = env.step(action)

        # Parse observation and create input for models
        obs_pre = transform.resize(next_obs, X_size)

        # Clip reward to +-0
        r = np.clip(reward, -1, 1)
        reward_accum += r

        # Update our experience pool
        # We need to have some kind of cache here so we can store the m most
        # recent experience tuples together
        experience = (obs, action, r, obs_pre, done)
        if t >= _EXP_REPLAY_FRAMES:
            _ = D.pop()
        D.append(experience)

        # Updating epsilon value after random filling of experience pool
        if t >= _EXP_REPLAY_FRAMES and t < _EXP_REPLAY_FRAMES*2:
            epsilon -= (.9)/_EXP_REPLAY_FRAMES
        elif t == _EXP_REPLAY_FRAMES*2:
            epsilon = .1

        if t >= _EXP_REPLAY_FRAMES:
            # Sample minibatch of transitions
            sample = sample_experience()
            X = np.concatenate(sample[:, 0], 0)
            Y = gen_y(sample)

            # Gradient descent
            DQN_estimate.update(X, Y)

            # Reset Q_target = Q_estimate
            if t+1 % M == 0:
                DQN_estimate.save_model(checkpoint)
                DQN_target.load_model(checkpoint)


        # Marks the end of an episode
        if done:
            print("Episode finished after {} timesteps.".format(t+1))
            obs_pre = transform.resize(env.reset(), X_size)
            action = 0
            reward = 0

            metrics.append((reward_accum/(t-_t), values_accum/(t-_t)))
            print("avg_reward over episode: {}".format(reward_accum/(t-_t)))
            print("avg_values over episode: {}".format(values_accum/(t-_t)))
            reward_accum = 0.0
            values_accum = 0.0

            _t = t

DQN_estimate.close()


