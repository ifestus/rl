import gym
from skimage import transform
import tensorflow as tf
import numpy as np

from dqn import DQN

env = gym.make('Pong-v0')
print(env.observation_space)
print(env.action_space)

sess = tf.Session()

exp_max_frames = 100000
D = []
X_size = (84, 84, 1)
# Number of frames our sample will take
m = 4

DQN_estimate = DQN(sess, name="estimate")
DQN_target   = DQN(sess, name="target")


# Prefill D

def sample_experience(minibatch_size=32):
    # Fill the sample with zeros to begin with
    sample = np.zeros((minibatch, X_size[0], X_size[1], X_size[2]))
    # We want to add ${minibatch} experience values to our samples
    for j in range(minibatch):
        # k is the index to our experience pool (D)
        k = np.randint(exp_max_frames)
        sample[j] = D[k]


for i_episode in range(20):
    obs = env.reset()
    reward = 0
    action = 0
    for t in range(1000):
        env.render()
        # Get action from the estimate network
        # this action is going to be an e-greedy one that is determined by the
        # estimate network
        obs = obs_pre

        DQN_estimate.predict(np.expand_dims(obs, 0))
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

        # Sample minibatch of transitions
        # Gradient descent
        # Reset Q_target = Q_estimate

        # Marks the end of an episode
        if done:
            print("Episode finished after {} timesteps.".format(t+1))
            break

DQN_estimate.close()


