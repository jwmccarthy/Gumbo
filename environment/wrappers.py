import numpy as np
import torch as th
import gymnasium as gym

from utils import transpose_dict, safe_from_numpy


class TorchIO(gym.Wrapper):
    """Converts numpy arrays to tensors on reset and step."""
    
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        """Convert initial obs to tensor on reset"""
        obs, info = super().reset(seed=seed, options=options)
        return th.from_numpy(obs), info
    
    def step(self, actions):
        """Convert next episode values to tensor on step"""
        if isinstance(actions, th.Tensor):
            actions = actions.cpu().detach().numpy()

        obs, rewards, terms, truncs, infos = super().step(actions)

        # handle data in infos
        infos = transpose_dict(infos)
        infos = [{k: safe_from_numpy(v) for k, v in d.items()} for d in infos]

        return (
            th.from_numpy(obs),
            th.from_numpy(rewards),
            th.from_numpy(terms),
            th.from_numpy(truncs),
            infos
        )