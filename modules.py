import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, activation=nn.Tanh):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.network = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )