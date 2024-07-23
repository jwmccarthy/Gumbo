import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.spaces import flatdim

from utils import transpose_dict, try_from_numpy
from environment.utils import logit_dim


class TorchIO(gym.Wrapper):
    """Converts numpy arrays to tensors on reset and step."""
    
    def __init__(self, env):
        super().__init__(env)

        # base attributes
        self.num_envs = self.unwrapped.num_envs
        self.single_action_space = self.unwrapped.single_action_space
        self.single_observation_space = self.unwrapped.single_observation_space

        # derived attributes
        self.flat_dim = flatdim(self.single_observation_space)
        self.logit_dim = logit_dim(self.single_action_space)

    def reset(self, seed=None, options=None):
        """Convert initial obs to tensor on reset"""
        obs, info = super().reset(seed=seed, options=options)
        return th.from_numpy(obs), info
    
    def step(self, actions):
        """Convert next episode values to tensor on step"""
        if isinstance(actions, th.Tensor):
            actions = actions.cpu().detach().numpy()

        obs, rewards, terms, truncs, infos = super().step(actions)

        # TODO: custom env structure to handle episode data better
        infos = transpose_dict(infos) or [{} for _ in range(self.num_envs)]
        infos = [{k: try_from_numpy(v) for k, v in d.items()} for d in infos]

        return (
            th.from_numpy(obs),
            th.from_numpy(rewards),
            th.from_numpy(terms),
            th.from_numpy(truncs),
            infos
        )