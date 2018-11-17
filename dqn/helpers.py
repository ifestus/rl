import numpy as np

def action_from_dqn(obs, num_replay_frames, estimate_dqn)


# params.obs {Object} - Observation from gym environment
# params.experience_replay {Object} - Experience replay object
# params.num_replay_frames {Int} - Size of replay memory
# params.estimate_dqn {Object} - Instance of Estimate DQN
# params.metrics {Object} - metrics object for logging
def action_from_dqn(params):
    obs = params.obs
    experience_replay = params.experience_replay
    num_replay_frames = params.num_replay_frames
    estimate_dqn = params.estimate_dqn
    metics = params.metrics

    valid_actions = estimate_dqn.get_valid_actions()

    # experience_replay_frames - num_replay_frames + 1 because we're going to grab
    # the most recent num_replay_frames+1 frames from the top of experience_replay
    # and append our observation to that
    X = np.concatenate([experience_replay[x][0] for x in range(experience_replay_frames-num_replay_frames+1,
                                                               experience_replay_frames, 1)],
                       -1)
    X = np.expand_dims(np.append(X, obs, -1), 0)
    action_values = estimate_dqn.action_values(X)
    action = np.argmax(action_values, 1)[0]

    metrics.add_to_values_accum(
        np.reshape(action_values, (valid_actions))[action])

    return action


def take_step(obs, t, action, gym_env, transform_size, experience_replay, metrics):
# params.obs {Object} - Observation object from gym environment
# params.experience_replay {Object} - Experience replay object
# params.num_replay_frames {Int} - 
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
        next_obs_transformed = transform.resize(next_obs, transform_size)

        r = np.clip(reward, -1, 1)
        if metrics:
            metrics.add_to_rewards_accum(r)

        experience = (obs, action, r, next_obs_transformed, done)
        if len(experience_replay) >= experience_replay_frame:
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
