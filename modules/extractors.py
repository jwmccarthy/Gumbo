import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLPExtractor(nn.Module):

    def __init__(self, in_dims, out_dims):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dims)
        )