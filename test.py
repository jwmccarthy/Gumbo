import torch as th
import torch.nn as nn
import gymnasium as gym
from environment.wrappers import TorchIO
from data.collector import Collector
from data.buffers.episodic import EpisodicBuffer
from policy.discrete import CategoricalPolicy
from policy.continuous import DiagonalGaussianPolicy
from learn.ppo import PPO


if __name__ == "__main__":
    env = gym.vector.SyncVectorEnv(1 * [lambda: gym.make("CartPole-v1")])
    env = TorchIO(env)

    policy = CategoricalPolicy(
        nn.Sequential(
            nn.Linear(env.flat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, env.logit_dim)
        )
    )

    buff = EpisodicBuffer(env, 2048)

    collector = Collector(env, policy, buff)

    ppo = PPO(policy, 
              nn.Sequential(
                    nn.Linear(env.flat_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
              ), collector)
    data = ppo.learn(int(1e4))

    # Example expert trajectories
env = gym.make("CartPole-v1", render_mode="human")
obs = th.as_tensor(env.reset()[0])
for i in range(10000):
    action = policy(obs, sample=False)
    obs, _, term, trunc, _ = env.step(action.detach().numpy())
    if term or trunc:
        obs = env.reset()[0]
    obs = th.as_tensor(obs)
    