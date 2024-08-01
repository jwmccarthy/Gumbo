import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import gymnasium as gym
from gumbo.environment.wrappers import TorchIO
from gumbo.data.collector import Collector
from gumbo.data.buffers import EpisodicBuffer
from gumbo.policy.discrete import CategoricalPolicy
from gumbo.policy.continuous import DiagonalGaussianPolicy
from gumbo.learn.ppo import PPO

import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = gym.vector.SyncVectorEnv(1 * [lambda: gym.make("LunarLander-v2")])
    env = TorchIO(env)

    dfs = []
    for i in range(5):
        policy = CategoricalPolicy(
            nn.Sequential(
                nn.Linear(env.flat_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, env.logit_dim)
            )
        )

        critic = nn.Sequential(
            nn.Linear(env.flat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        buff = EpisodicBuffer(env, 2048)

        collector = Collector(env, policy, buff)

        ppo = PPO(policy, critic, collector)
        rews, lens = ppo.learn(int(1e5))
        rews = np.array(rews)
        lens = np.array(lens)

        n = 20
        sum_ep_lens = np.cumsum(lens)

        dfs.append(pd.DataFrame({
            "reward": rews,
            "timestep": np.round(sum_ep_lens, -3),
            "run": i
        }))

    data = pd.concat(dfs)

    plt.figure()
    sns.lineplot(x="timestep", y="reward", errorbar="sd", estimator="mean", data=data, err_kws={"edgecolor": None})
    plt.show()