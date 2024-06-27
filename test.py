import numpy as np
import torch as th
import gymnasium as gym
from environment.wrappers import TorchIO
from buffers.episodic import EpisodicBuffer


if __name__ == "__main__":
    env = gym.vector.SyncVectorEnv(3 * [lambda: gym.make("CartPole-v1")])
    env = TorchIO(env)

    buff = EpisodicBuffer(env, 2048)

    obs, infos = env.reset()

    for _ in range(2048):
        actions = env.action_space.sample()
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        buff.add(obs, th.from_numpy(actions), rewards, terms, truncs, infos)

    for e in buff.episodes:
        print(e.env_idx, e.indices, e.trunc, e.final_observation, len(e))

    print(len(buff.episodes))