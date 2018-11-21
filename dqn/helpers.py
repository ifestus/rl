import numpy as np


# params.obs {Object} - Observation from gym environment
# params.experience_replay {Object} - Experience replay object
#   the underestanding is that len(experience_replay) returns
#   experience_replay_max_size (it's full)
# params.m {int} - Analogous to the m value
#   number of previous frames to include in the input to DQN
# params.estimate_dqn {Object} - Instance of Estimate DQN
# params.metrics {Object} - metrics object for logging
# returns {Object} np.array - Input object to DQN with size of
#   (obs.shape[0], obs.shape[1], m+1)
def craft_input_with_replay_memory(params):
    obs = params.obs
    experience_replay = params.experience_replay
    m = params.m
    metrics = params.metrics

    # len(experience_replay) - m + 1 because we're going to grab
    # the most recent m+1 frames from the top of experience_replay
    # and append our observation to that
    # NOTE: This function should never be called unless len(experience_replay) == experience_replay_max_size
    X = np.concatenate([experience_replay[x][0] for x in range(len(experience_replay)-m+1,
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


# params.minibatch_size {int} - size of the minimatch
# params.experience_replay {Object} - experience replay object
#   the underestanding is that len(experience_replay) returns
#   experience_replay_max_size (it's full)
# params.m {int} - Analogous to the m value
#   number of previous frames to include in the input to DQN
def sample_from_experience(params):
    minibatch_size = params.minibatch_size
    experience_replay = params.experience_replay
    m = params.m

    sample = {'st': [], 'at': [], 'rt+1': [], 'st+1': [], 'dt+1': []}

    # We want to add ${params.minibatch_size} experience tuples to our sample
    for _ in range(minibatch_size):
        # k is th eindex into our experience_replay
        # we don't want to start our sample with a done frame because that
        # makes things a bit more complicated.
        k = np.maximum(np.randint(len(experience_replay), m-1))

        # If any of the frames from k-m+1 to k-1 (inclusive) are done frames
        if np.amax(experience_replay[k-m+1:k][1][4]) is True:
            done_frame = np.argmax(experience_replay[k-m+1:k][1][4])
            new_k = k - done_frame

            # shift k such that the done frame is new k
            if k - done_frame > m-1:
                k = new_k
            else:
                k = np.maximum(np.random.randint(len(experience_replay), m-1))

        st = []
        at = []
        rt1 = []
        st1 = []
        dt1 = 0

        for x in range(k, k-m, -1):
            # st and st1 will each have to be np.concatenated to create a
            # (84, 84, m)-shape np array
            st.insert(0, experience_replay[x][0])
            st1.insert(0, experience_replay[x][3])

            at.insert(0, experience_replay[x][1])
            rt1.insert(0, experience_replay[x][2])

            dt1 = dt1 or experience_replay[x][4]

        sample['st'].append(np.concatenate(st, -1))
        sample['at'].append(at)
        sample['rt+1'].append(rt1)
        sample['st+1'].append(np.concatenate(st1, -1))
        sample['dt+1'].append(dt1)

    # Lets turn those pesky lists in sample into np arrays
    for key in sample:
        sample[key] = np.array(sample[key])

    return sample


# sample {Object} - sample returned from sample_from_experience fn
# gamma {int} - decay paramter
# target_dqn {Object} - target dqn object
# minibatch_size {int} - size of the minibatch
def gen_y(sample, gamma, target_dqn, minibatch_size):
    rt1 = sample['rt+1'][:][-1]
    done = sample['dt+1']

    # Input to the target DQN is t+1 state
    X = sample['st+1']

    Y = rt1 + gamma * np.multiply(done,
                                   np.amax(target_dqn.action_values(X), -1))

    return np.reshape(Y, (minibatch_size, 1))


def update_models_from_experience(t, experience_replay_max_frames, estimate_dqn, target_dqn):
    # Update models after we fill experience replay and only on every 4th frame
    if t >= experience_replay_max_frames and t+1 % 4 == 0:
        sample = sample_from_experience()
        X = sample['st']
        Y = gen_y(sample)

        if (t+1) % 12000 == 0:
            print('Y for sampled values: ', Y)

        # Update estimate model
        estimate_dqn.update(X, Y)

        # Reset Q_TARGET = Q_ESTIMATE
        if t+1 % _C == 0:
            estimate_dqn.save_model(_CHECKPOINT_FILE)
            target_dqn.load_model(_CHECKPOINT_FILE)

