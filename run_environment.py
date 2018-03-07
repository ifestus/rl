import gym
from skimage import transform
import tensorflow as tf
import numpy as np

from dqn import DQN

env = gym.make('Pong-v0')
print(env.observation_space)
print(env.action_space)

sess = tf.Session()

_TRAIN_FRAMES = 1000000
_EXP_REPLAY_FRAMES = 100000

D = []
X_size = (84, 84, 1)
# Number of previous frames we'll take to create a sample for input to DQN
m = 3
epsilon = 1.0

DQN_estimate = DQN(sess, name="estimate")
DQN_target   = DQN(sess, name="target")


# Prefill D
# D = [(np.zeros((84, 84, 1)), 0, 0, np.zeros((84, 84, 1)))] * _EXP_REPLAY_FRAMES

def sample_experience(minibatch_size=32):
    # Fill the sample with zeros to begin with
    sample = []
    # sample = np.zeros((minibatch_size, X_size[0], X_size[1], X_size[2]))

    # We want to add ${minibatch} experience values to our samples
    for _ in range(minibatch_size):
        # k is the index to our experience pool (D)
        k = np.maximum(np.random.randint(_EXP_REPLAY_FRAMES), m)

        # Pull processed frames from the previous ${m-1} experiences
        # and add them to the sample =)
        a = np.concatenate([D[x][0] for x in range(k-m, k+1, 1)], -1)
        sample.append(a)

    return sample

# Really, we're training based on frames, so we should control the flow as such
for episode in range(1):
    obs_pre = env.reset()
    action = 0
    reward = 0
    for t in range(_TRAIN_FRAMES + 2*_EXP_REPLAY_FRAMES):
        if t >= _EXP_REPLAY_FRAMES:
            env.render()

        # Get action from the estimate network
        # this action is going to be an e-greedy one that is determined by the
        # estimate network
        obs = obs_pre

        # e-greedy action selection
        _ = np.random.normal()
        if _ > epsilon:
            action = DQN_estimate.predict(np.expand_dims(obs, 0))
        else:
            action = env.action_space.sample()

        # take action and preprocess next state
        next_obs, reward, done, info = env.step(action)

        # Parse observation and create input for models
        obs_pre = transform.resize(next_obs, X_size)

        # Clip reward to +-0
        r = np.clip(reward, -1, 1)

        # Update our experience pool
        # We need to have some kind of cache here so we can store the m most
        # recent experience tuples together
        experience = (obs, action, r, obs_pre)
        _ = D.pop()
        D.append(experience)

        # Updating epsilon value after random filling of experience pool
        if t >= _EXP_REPLAY_FRAMES and t < _EXP_REPLAY_FRAMES*2:
            epsilon -= (.9)/_TRAIN_FRAMES
        elif t == _EXP_REPLAY_FRAMES*2:
            epsilon = .1

        # Sample minibatch of transitions
        # Gradient descent
        # Reset Q_target = Q_estimate

        # Marks the end of an episode
        if done:
            print("Episode finished after {} timesteps.".format(t+1))
            break

DQN_estimate.close()


