import gym

env = gym.make('Pong-v0')
print(env.observation_space)
print(env.action_space)

for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print("Observation: ", observation)
        # this is where my code goes -- doing da actions man
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps.".format(t+1))
            break

