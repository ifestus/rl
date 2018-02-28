import gym
import tensorflow as tf

from dqn import DQN

env = gym.make('Pong-v0')
print(env.observation_space)
print(env.action_space)

sess = tf.Session()

DQN_behavior = DQN(sess, name="behavior")
DQN_target   = DQN(sess, name="target")

for i_episode in range(20):
    observation = env.reset()
    print("Observation: ", observation)
    for t in range(1000):
        # Parse observation and create input for models

        # Clip reward to +-0

        exit()
        env.render()
        # Get action from the behavior network
        # this action is going to be an e-greedy one that is determined by the
        # behavior network
        action = env.action_space.sample()

        # take action
        observation, reward, done, info = env.step(action)

        # Marks the end of an episode
        if done:
            print("Episode finished after {} timesteps.".format(t+1))
            break

def prepare_obs(observation):
    pass

