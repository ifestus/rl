import gym
from skimage import transform
import tensorflow as tf
import numpy as np

from dqn import DQN

env = gym.make('Pong-v0')
print(env.observation_space)
print(env.action_space)

sess = tf.Session()

D = [None] * 1000000
X_size = (84, 84, 1)

DQN_behavior = DQN(sess, name="behavior")
DQN_target   = DQN(sess, name="target")


for i_episode in range(20):
    observation = env.reset()
    reward = 0
    for t in range(1000):
        # Parse observation and create input for models
        obs = transform.resize(observation, X_size)

        # Clip reward to +-0
        r = np.clip(reward, -1, 1)

        env.render()
        # Get action from the behavior network
        # this action is going to be an e-greedy one that is determined by the
        # behavior network
        obs = np.concatenate([obs, obs, obs, obs], -1)

        DQN_behavior.predict(np.expand_dims(obs, 0))
        action = env.action_space.sample()

        # take action
        observation, reward, done, info = env.step(action)

        # Marks the end of an episode
        if done:
            print("Episode finished after {} timesteps.".format(t+1))
            break

DQN_behavior.close()


