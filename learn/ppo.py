from itertools import chain

import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from data.samplers import batch_sample


class PPO:

    def __init__(
        self, 
        policy, 
        critic, 
        collector, 
        optimizer=Adam,
        epochs=10,
        buff_size=2048, 
        batch_size=64,
        eps=0.2,
        gamma=0.99,
        lmbda=0.95,
        ent_coef=0.0,
        val_coef=0.5,
        max_norm=0.5,
        opt_kwargs={}
    ):
        self.policy = policy
        self.critic = critic
        self.collector = collector

        # data hyperparams
        self.epochs = epochs
        self.buff_size = buff_size
        self.batch_size = batch_size

        # training hyperparams
        self.eps = eps
        self.gamma = gamma
        self.lmbda = lmbda
        self.ent_coef = ent_coef
        self.val_coef = val_coef
        self.max_norm = max_norm

        # initialize optimizer for combined params
        self.params = chain(self.policy.parameters(),
                            self.critic.parameters())
        self.optimizer = optimizer(self.params, **opt_kwargs)

    def learn(self, steps):
        for _ in range(0, steps, self.buff_size):
            buffer = self.collector.collect(self.buff_size)
            data = self._augment_training_data(buffer).flatten()

            for _ in range(self.epochs):
                for batch in batch_sample(data, self.batch_size):
                    # evaluate obs, actions
                    values = self.critic(batch.obs).squeeze()
                    dist = self.policy.dist(batch.obs)
                    log_probs = dist.log_prob(batch.act)
                    entropy = dist.entropy()

                    # policy loss
                    ratios = th.exp(log_probs - batch.lgp)
                    advantages = self._normalize_advantages(batch.adv)
                    policy_loss = -th.min(
                        advantages * ratios,
                        advantages * th.clamp(ratios, 1 - self.eps, 1 + self.eps)
                    ).mean()

                    # value loss
                    value_loss = F.mse_loss(values, batch.val)

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
            ep_data.adv[:] = self._calculate_episode_advantages(
                ep_data.rew, ep_data.val, final_value
            )

        return data
    
    def _calculate_episode_advantages(self, rewards, values, final_value):
        next_vals = th.cat([values[1:], final_value])
        td_errors = (rewards + self.gamma * next_vals - values).view(1, 1, -1)

        T = len(rewards)

        # conv kernel of discount factors
        kernel = (self.gamma * self.lmbda) ** th.arange(T, dtype=th.float32).view(1, 1, -1)

        return F.conv1d(td_errors, kernel, padding=T - 1).squeeze()[-T:]
    
    def _normalize_advantages(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)