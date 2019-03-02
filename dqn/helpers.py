import numpy as np
from skimage import transform


# params.obs {Object} - Observation from gym environment
# params.experience_replay {Object} - Experience replay object
#   the underestanding is that len(experience_replay) returns
#   experience_replay_max_frames (it's full)
# params.m {int} - Analogous to the m value
#   number of previous frames to include in the input to DQN
# params.estimate_dqn {Object} - Instance of Estimate DQN
# params.metrics {Object} - metrics object for logging
# returns {Object} np.array - Input object to DQN with size of
#   (obs.shape[0], obs.shape[1], m+1)
def craft_input_with_replay_memory(params):
    obs = params['obs']
    experience_replay = params['experience_replay']
    m = params['m']

    # len(experience_replay) - m + 1 because we're going to grab
    # the most recent m+1 frames from the top of experience_replay
    # and append our observation to that
    # NOTE: This function should never be called unless len(experience_replay) == experience_replay_max_frames
    X = np.concatenate([experience_replay[x][0] for x in range(len(experience_replay)-m+1,
                                                               len(experience_replay), 1)],
                       -1)
    X = np.expand_dims(np.append(X, obs, -1), 0)
    return X


# input_object {Object} - object returned from craft_input_with_replay_memory fn
# estimate_dqn {Object} - DQN used for estimation
# metrics {Object} - Object for writing to metrics
def action_from_dqn(input_object, estimate_dqn, metrics):
    valid_actions = estimate_dqn.get_valid_actions()
    action_values = estimate_dqn.action_values(input_object)
    action = np.argmax(action_values, 1)[0]

    metrics.add_to_values_accum(
        np.reshape(action_values, (valid_actions))[action])

    return action


# params.action {int} - number of the action to take
# params.t {int} - step number
# params.obs {Object} - Observation object from gym environment
# params.done {Bool} - whether or not the episode is done - returned by gym_env.step
# params.gym_env {Object} - Gym environment object
# params.experience_replay {Object} - Experience replay object
# params.experience__replay_max_frames {int} - Max size of the experience replay
# params.transform_size {int} - shape to transform observation to (i.e. [84, 84]) or False
# params.metrics {Object} - Object for writing to metrics
def take_step(params):
    done = params['done']
    t = params['t']
    action = params['action']
    gym_env = params['gym_env']
    transform_size = params['transform_size']
    metrics = params['metrics']

    if not done:
        next_obs, reward, done, info = gym_env.step(action)
    else:
        next_obs, reward, done, info = gym_env.reset()
        _t = metrics.get_episodic_t()
        print('Episode finished after {} timesteps ({}).'.format(t+1, _t))

        metrics.write_metrics()
        metrics.reset_to_episode_start()

    # needs to be a step here that takes the max pixel value between frame n and n-1
    next_obs_transformed = transform.resize(next_obs, transform_size) if transform_size else next_obs

    r = np.clip(reward, -1, 1)
    if metrics:
        metrics.add_to_rewards_accum(r)

    return next_obs_transformed, r, done, info

def append_to_experience_replay(params):
    prev_obs = params['prev_obs']
    action = params['action']
    reward = params['reward']
    done = params['done']
    next_obs_transformed = params['next_obs_transformed']
    experience_replay = params['experience_replay']
    experience_replay_max_frames = params['experience_replay_max_frames']

    # Add this observation to experience replay
    experience = (prev_obs, action, reward, next_obs_transformed, done)
    if len(experience_replay) >= experience_replay_max_frames:
        experience_replay.pop(0)
    experience_replay.append(experience)

    return


# params.minibatch_size {int} - size of the minimatch
# params.experience_replay {Object} - experience replay object
#   the underestanding is that len(experience_replay) returns
#   experience_replay_max_frames (it's full)
# params.m {int} - Analogous to the m value
#   number of previous frames to include in the input to DQN
def sample_from_experience(params):
    minibatch_size = params['minibatch_size']
    experience_replay = params['experience_replay']
    m = params['m']

    sample = {'st': [], 'at': [], 'rt+1': [], 'st+1': [], 'dt+1': []}

    # We want to add ${params.minibatch_size} experience tuples to our sample
    for _ in range(minibatch_size):
        # k is th eindex into our experience_replay
        # We will be backfilling from k, so we'd like to be at least m-1 frames away from
        # the beginning
        k = np.maximum(np.random.randint(len(experience_replay)), m-1)

        # we don't want to start our sample with a done frame because that
        # makes things a bit more complicated.
        # If any of the frames from k-m+1 to k-1 (inclusive) are done frames
        if np.amax(experience_replay[k-m+1:k][1][4]) is True:
            done_frame = np.argmax(experience_replay[k-m+1:k][1][4])
            new_k = k - m + 1 + done_frame

            # shift k such that the done frame is new k if new_k is far enough from beginning
            if new_k > m-1:
                k = new_k

            # Otherwise just get a new k - this might have a done frame within it, but
            # we should only very, very rarely get here
            else:
                k = np.maximum(np.random.randint(len(experience_replay)), m-1)

        st = []  # State at time t
        at = []  # Action at time t
        rt1 = []  # Reward at time t+1
        st1 = []  # State at time t+1
        dt1 = 0  # Done at t+1

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


# t {int} - time step integer
# experience_replay_max_frames {int} - max size of experience replay
# estimate_dqn {Object} - estimate dqn
# target_dqn {Object} - target dqn
# checkpoint_file {string} - checkpoint file name
# update_freq {int} - update frequency value from paper
# minibatch_size {int} - size of minibatch
# m {int} - Agent history length - number of frames fed into DQN model
def update_models_from_experience(params):
    t = params['t']
    gamma = params['gamma']
    minibatch_size = params['minibatch_size']
    estimate_dqn = params['estimate_dqn']
    target_dqn = params['target_dqn']
    checkpoint_file = params['checkpoint_file']
    # Update models after we fill experience replay and only on every 4th frame
    if t >= params['experience_replay_max_frames'] and t+1 % 4 == 0:
        sample = sample_from_experience(params)
        X = sample['st']
        Y = gen_y(sample, gamma, target_dqn, minibatch_size)

        if (t+1) % 12000 == 0:
            print('Y for sampled values: ', Y)

        # Update estimate model
        estimate_dqn.update(X, Y)

        # Reset Q_TARGET = Q_ESTIMATE
        if t+1 % params['update_frequency'] == 0:
            estimate_dqn.save_model(checkpoint_file)
            target_dqn.load_model(checkpoint_file)
