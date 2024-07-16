import torch as th
import gymnasium as gym
from environment.wrappers import TorchIO
from data.collector import Collector
from data.buffers.episodic import EpisodicBuffer
from policy.discrete import CategoricalPolicy
from policy.continuous import DiagonalGaussianPolicy


if __name__ == "__main__":
    env = gym.vector.SyncVectorEnv(3 * [lambda: gym.make("LunarLander-v2")])
    env = TorchIO(env)

    agent = CategoricalPolicy(th.nn.Linear(env.flat_dim, env.logit_dim))

    buff = EpisodicBuffer(env, 2048)

    collector = Collector(env, agent, buff)

    data = collector.collect(2048)

    for e in buff.episodes:
        print(e.env_idx, e.indices, e.trunc, len(e))













    