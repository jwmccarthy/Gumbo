import torch as th
import torch.nn as nn
from torch.distributions import MultivariateNormal


class DiagonalGaussianPolicy(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.covmat = th.eye(model[-1].out_features)

    def forward(self, obs, sample=True):
        loc = self.model(obs)
        dist = MultivariateNormal(loc=loc, covariance_matrix=self.covmat)
        return dist.sample() if sample else loc

    def dist(self, obs):
        loc = self.model(obs)
        return MultivariateNormal(loc=loc, covariance_matrix=self.covmat)