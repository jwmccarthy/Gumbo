import gymnasium as gym


def logit_dim(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        return action_space.shape[0]
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return sum(action_space.nvec)
    else:
        raise NotImplementedError("Action space not supported")