from itertools import chain

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from gumbo.data.samplers import BatchSampler


class PPO:

    def __init__(
        self, 
        policy, 
        critic, 
        collector, 
        optimizer=Adam,
        epochs=10,
        batch_size=64,
        lr=3e-4,
        eps=0.2,
        gamma=0.99,
        lmbda=0.95,
        ent_coef=0.01,
        val_coef=0.5,
        max_norm=0.5
    ):
        self.policy = policy
        self.critic = critic
        self.collector = collector

        # data hyperparams
        self.epochs = epochs
        self.batch_size = batch_size

        # training hyperparams
        self.lr = lr
        self.eps = eps
        self.gamma = gamma
        self.lmbda = lmbda
        self.ent_coef = ent_coef
        self.val_coef = val_coef
        self.max_norm = max_norm

        # initialize optimizer for combined params
        self.params = chain(self.policy.parameters(),
                            self.critic.parameters())
        self.optimizer = optimizer(self.params, lr=self.lr)

    def learn(self, steps):
        ep_rew = []
        ep_len = []

        for _ in range(0, steps, self.collector.buffer.size):
            buffer = self.collector.collect()

            data = self._augment_training_data(buffer)

            for episode in buffer.episodes:
                indices = episode.indices
                env_idx = episode.env_idx
                ep_data = data[indices, env_idx]
                ep_rew.append(ep_data.rew.sum().item())
                ep_len.append(len(ep_data))

            print(np.mean(ep_rew[-100:]))

            sampler = BatchSampler(data, self.batch_size, self.epochs)

            while batch := sampler.sample():
                self._update(batch)

            buffer.reset()

        return ep_rew, ep_len

    def _update(self, batch):
        # evaluate obs, actions
        values = self.critic(batch.obs).squeeze()
        dist = self.policy.dist(batch.obs)
        log_probs = dist.log_prob(batch.act)
        entropy = dist.entropy()

        # kl divergence
        kl_div = th.mean(batch.lgp - log_probs)

        # policy loss
        ratios = th.exp(log_probs - batch.lgp)
        advantages = self._normalize_advantages(batch.adv)
        policy_loss = -th.min(
            advantages * ratios,
            advantages * th.clamp(ratios, 1 - self.eps, 1 + self.eps)
        ).mean()

        # value loss
        value_loss = F.mse_loss(batch.ret.squeeze(), values)

        # entropy loss
        entropy_loss = -entropy.mean()

        # total loss
        total_loss = policy_loss \
                   + self.val_coef * value_loss \
                   + self.ent_coef * entropy_loss
        
        # gradients
        self.optimizer.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.params, self.max_norm)

        self.optimizer.step()
                    
    @th.no_grad()
    def _augment_training_data(self, buffer):
        data = buffer.get_data()

        # calculate action log probs
        dist = self.policy.dist(data.obs)
        data.lgp = dist.log_prob(data.act)
        
        # calculate values
        data.val = self.critic(data.obs).squeeze(2)

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
            ep_data.adv[:] = self._calculate_episode_advantages(
                ep_data.rew, ep_data.val, final_value
            )

        data.ret = data.val + data.adv

        return data
    
    def _calculate_episode_advantages(self, rewards, values, final_value):
        next_vals = th.cat([values[1:], final_value])
        td_errors = (rewards + self.gamma * next_vals - values).view(1, 1, -1)

        T = len(rewards)

        # conv kernel of discount factors
        kernel = (self.gamma * self.lmbda) ** th.arange(T, dtype=th.float32).view(1, 1, -1)

        return F.conv1d(td_errors, kernel, padding=T - 1).view(-1)[-T:]
    
    def _normalize_advantages(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)