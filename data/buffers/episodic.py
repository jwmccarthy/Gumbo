import numpy as np
import torch as th

from data.datasets import TensorDataset, Subset


class EpisodicBuffer:

    def __init__(self, env, size):
        self.env = env
        self.size = size
        self.num_env = self.env.num_envs
        self.obs_dim = env.single_observation_space.shape
        self.act_dim = env.single_action_space.shape

        self.episodes = []
        self.data = TensorDataset(
            obs=th.empty((self.size, self.env.num_envs, *self.obs_dim)),
            act=th.empty((self.size, self.env.num_envs, *self.act_dim)),
            rew=th.empty((self.size, self.env.num_envs))
        )

        self.start = self.num_env * [0]
        self.end = -1

    def add(self, obs, act, rew, term, trunc, info=None):
        self.data[self.end] = (obs, act, rew)
        self.end = (self.end + 1) % self.size
        for i in range(self.num_env):
            if term[i] or trunc[i]: self._create_episode(i, trunc, info)

    def _create_episode(self, env_idx, trunc, info):
        indices = slice(self.start[env_idx], self.end)
        episode = Subset(self.data, indices, env_idx=env_idx,
                         trunc=trunc[env_idx], **info[env_idx])
        self.start[env_idx] = self.end
        self.episodes.append(episode)

    def get_data(self):
        return self.data[:self.end+1]