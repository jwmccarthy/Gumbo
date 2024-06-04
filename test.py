import torch as th

import gymnasium as gym
from env.torch_env import SyncTorchEnv


if __name__ == "__main__":
    env = SyncTorchEnv("LunarLander-v2", 3)
    