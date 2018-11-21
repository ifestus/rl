import numpy as np


# params.obs {Object} - Observation from gym environment
# params.experience_replay {Object} - Experience replay object
# params.num_replay_frames {int} - Analogous to the m value
#   number of previous frames to include in the input to DQN
# params.estimate_dqn {Object} - Instance of Estimate DQN
# params.metrics {Object} - metrics object for logging
# returns {Object} np.array - Input object to DQN with size of
#   (obs.shape[0], obs.shape[1], num_replay_frames+1)
def craft_input_with_replay_memory(params):
    obs = params.obs
    experience_replay = params.experience_replay
    num_replay_frames = params.num_replay_frames
    metrics = params.metrics

    # len(experience_replay) - num_replay_frames + 1 because we're going to grab
    # the most recent num_replay_frames+1 frames from the top of experience_replay
    # and append our observation to that
    # NOTE: This function should never be called unless len(experience_replay) == experience_replay_max_size
    X = np.concatenate([experience_replay[x][0] for x in range(len(experience_replay)-num_replay_frames+1,
                                                               len(experience_replay), 1)],
                       -1)
    X = expand_dims(np.append(X, obs, -1), 0)
    return X


# input_object {Object} - object returned from craft_input_with_replay_memory fn
# estimate_dqn {Object} - DQN used for estimation
# metrics {Object} - Object for writing to metrics
def action_from_dqn(input_object, estimate_dqn, metrics):
    valid_actions = estimate_dqn.get_valid_actions()
    action_values = estimate_dqn.action_values(X)
    action = np.argmax(action_values, 1)[0]

    metrics.add_to_values_accum(
        np.reshape(action_values, (valid_actions))[action])

    return action


# params.action {int} - number of the action to take
# params.t {int} - step number
# params.obs {Object} - Observation object from gym environment
# params.gym_env {Object} - Gym environment object
# params.experience_replay {Object} - Experience replay object
# params.num_replay_max_size {int} - Max size of the experience replay
# params.transform_size {int} - shape to transform observation to (i.e. [84, 84])
# params.metrics {Object} - Object for writing to metrics
def take_step(params):
    obs = params.obs
    t = params.t
    action = params.action
    gym_env = params.gym_env
    transform_size = params.transform_size
    experience_replay = params.experience_replay
    metrics = params.metrics

    next_obs, reward, done, info = gym_env.step(action)
    if not done:
        # needs to be a step here that takes the max pixel value between frame n and n-1
        next_obs_transformed = transform.resize(next_obs, transform_size)

        r = np.clip(reward, -1, 1)
        if metrics:
            metrics.add_to_rewards_accum(r)

        experience = (obs, action, r, next_obs_transformed, done)
        if len(experience_replay) >= experience_replay_max_size:
            experience_replay.pop(0)
        experience_replay.append(experience)

    else:
        next_obs_transformed = transform.resize(gym_env.reset(),
                                                     transform_size)

        if metrics:
            _t = metrics.get_episodic_t()
            print('Episode finished after {} timesteps ({}).'.format(t+1, _t))

            metrics.write_metrics()
            metrics.reset_to_episode_start()

    return next_obs_transformed
