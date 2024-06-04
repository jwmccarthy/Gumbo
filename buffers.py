import attrs
import numpy as np
import torch as th

from data.datasets import NamedTensorDataset


@attrs.define()
class Episode:
    env_idx:  slice      # index of environment
    mem_idx:  slice      # location in shared buffer memory
    obs:      th.Tensor
    act:      th.Tensor
    rew:      th.Tensor
    trunc:    bool       # flag indicating episode truncation
    last_obs: th.Tensor  # location following episode truncation


class EpisodicReplayBuffer:

    def __init__(self, env, max_size=1024):
        self.episodes = []
        self.step_idx = -1
        self.ep_start = np.zeros(env.num_env)

        # data dims
        self.max_size = max_size
        self.num_envs = self.env.num_envs
        self.batch_dim = (self.max_size, self.num_envs)
        
        # tensor storage for shared replay memory
        self.data = NamedTensorDataset(
            obs=th.zeros(self.batch_dim + env.obs_dim),
            act=th.zeros(self.batch_dim + env.act_dim),
            rew=th.zeros(self.batch_dim)
        )

    def add(self, obs, act, rew, trunc=None, last_obs=None):
        # circular indexing
        self.step_idx %= self.max_size + 1

        # save transition to buffer
        self.data[self.step_idx] = (obs, act, rew)

        # create episodes from terminal transitions
        if trunc and last_obs:
            for i in range(self.num_envs): 
                self._create_episode(i, last_obs[i], trunc[i])
            

    def _create_episode(self, env_idx, last_obs, trunc):
        if not last_obs: return  # exit if no last obs

        # obtain shared memory
        mem_idx = slice(self.ep_start[env_idx], self.step_idx+1)
        ep_data = self.data[mem_idx]

        self.episodes.append(Episode(env_idx,
                                     mem_idx,
                                     ep_data.obs,
                                     ep_data.act,
                                     ep_data.rew,
                                     last_obs,
                                     trunc))

        # set new episode start offset
        self.ep_start[env_idx] = self.step_idx