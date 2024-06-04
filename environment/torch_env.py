import torch as th
import gymnasium as gym


class SyncTorchEnv(gym.vector.SyncVectorEnv):

    def __init__(self, env_id, num_envs, make_args={}, init_args={}):
        self.env_id = env_id
        self.num_envs = num_envs

        # initialize parent SyncVectorEnv
        super().__init__(env_fns=self.num_envs * [
            lambda: gym.make(self.env_id, **make_args)
        ], **init_args)

    def reset(self, seed=None, options=None):
        """Convert initial obs to tensor on reset"""
        obs, info = super().reset(seed=seed, options=options)
        return th.from_numpy(obs), info
    
    def step(self, actions):
        """Convert next episode values to tensor on step"""
        if isinstance(actions, th.Tensor):
            actions = actions.cpu().detach().numpy()

        obs, rewards, terms, truncs, infos = super().step(actions)

        return (
            th.from_numpy(obs),
            th.from_numpy(rewards),
            th.from_numpy(terms),
            th.from_numpy(truncs),
            infos
        )