import numpy as np
import torch as th
import gymnasium as gym
from environment.wrappers import TorchIO
from data.buffers.episodic import EpisodicBuffer


if __name__ == "__main__":
    env = gym.vector.SyncVectorEnv(3 * [lambda: gym.make("BipedalWalker-v3")])
    env = TorchIO(env)

    buff = EpisodicBuffer(env, 2048)

    obs, infos = env.reset()

    for _ in range(1024):
        actions = env.action_space.sample()
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        buff.add(obs, th.from_numpy(actions), rewards, terms, truncs, infos)

    for e in buff.episodes:
        print(e.env_idx, e.indices, e.trunc, len(e))

    print(buff.get_data().obs[3])

    print(env)

    print(env.logit_dim, env.flat_dim)