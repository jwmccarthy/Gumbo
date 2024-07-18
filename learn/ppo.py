from itertools import chain

import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from data.datasets import NamedTensorDataset


class PPO:

    def __init__(
        self, 
        policy, 
        critic, 
        collector, 
        optimizer=Adam, 
        opt_kwargs={}
    ):
        self.policy = policy
        self.critic = critic
        self.collector = collector

        # initialize optimizer for combined params
        self.optimizer = optimizer(chain(policy.parameters(), 
                                         critic.parameters()),
                                   **opt_kwargs)

        self.gamma = 0.99
        self.lmbda = 0.95

    def learn(self, steps):
        buffer = self.collector.collect(2048)
        return self._augment_training_data(buffer)

    def _augment_training_data(self, buffer):
        data = buffer.get_data()

        # calculate action log probs
        dist = self.policy.dist(data.obs)
        data.lgp = dist.log_prob(data.act)
        
        # calculate values
        data.val = self.critic(data.obs).squeeze()

        # advantage storage
        data.adv = th.empty_like(data.val)

        # calculate adv. episode-wise
        for episode in buffer.episodes:
            env_idx = episode.env_idx
            indices = episode.indices
            ep_data = data[indices, env_idx]

            # calculate terminal obs value
            final_value = self.critic(episode.final_observation) * episode.trunc

            # calculate advantages
            advantages = self._calculate_episode_advantages(
                ep_data.rew, ep_data.val, final_value
            )
            
            # store calculated data
            data.adv[indices, env_idx] = advantages

        data.ret = data.val + data.adv

        return data

    def _calculate_episode_advantages(self, rewards, values, final_value):
        T = len(rewards)
        next_vals = th.cat([values[1:], final_value])
        td_errors = (rewards + self.gamma * next_vals - values).view(1, 1, -1)

        # conv kernel of discount factors
        kernel = (self.gamma * self.lmbda) ** th.arange(T, dtype=th.float32).view(1, 1, -1)

        # Perform the convolution
        return F.conv1d(td_errors, kernel, padding=T - 1).squeeze()[-T:]