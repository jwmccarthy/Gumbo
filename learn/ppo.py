from itertools import chain

import torch as th
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

        # add relevant tensors
        new_data = NamedTensorDataset(
            lgp=th.empty_like(data.rew),
            val=th.empty_like(data.rew),
            adv=th.empty_like(data.rew)
        )

        # calculate values episode-wise
        for episode in buffer.episodes:
            env_idx = episode.env_idx
            indices = episode.indices
            ep_data = data[indices, env_idx]

            # calculate log probs
            dist = self.policy.dist(ep_data.obs)
            log_probs = dist.log_prob(ep_data.act)

            # calculate values
            values = self.critic(ep_data.obs)
            final_value = self.critic(episode.final_observation)

            # calculate advantages
            advantages = self._calculate_episode_advantages(
                ep_data.rew, values, episode.trunc, final_value
            )

            print(new_data[indices, env_idx].lgp.shape)

            # store calculated data
            new_data[indices, env_idx] = (
                log_probs, values, advantages
            )

        # calculate returns
        data.ret = data.adv + data.val

        return data + new_data

    def _calculate_episode_advantages(self, rew, val, trunc, final_value):
        next_val = trunc and final_value
        advantages = th.zeros_like(rew)

        next_adv = 0.0
        for i in reversed(range(len(rew))):
            adv = rew[i] + self.gamma * next_val \
                         - val[i] \
                         + self.gamma * self.lmbda * next_adv

            advantages[i] = adv
            next_val, next_adv = val[i], adv

        return advantages