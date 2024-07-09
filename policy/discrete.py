import torch as th
import torch.nn as nn
from torch.distributions import Categorical


class CategoricalPolicy(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, sample=True):
        logits = self.model(obs)
        dist = Categorical(logits=logits)
        return dist.sample() if sample else th.argmax(logits, dim=-1)

    def dist(self, obs):
        logits = self.model(obs)
        return Categorical(logits=logits)