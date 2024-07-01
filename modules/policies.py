import torch as th
import torch.nn as nn
import torch.nn.functional as F

from extractors import MLPExtractor
from environment.utils import flat_dim, logit_dim


class Policy(nn.Module):

    def __init__(self, env, extractor=MLPExtractor):
        super().__init__()

        self.logit_dim = logit_dim(env)
        self.extractor = extrator()

        self.policy = nn.Sequential(
            self.extractor
        )