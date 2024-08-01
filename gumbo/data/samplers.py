from itertools import chain
from collections import deque
from numpy.random import permutation

from gumbo.utils import array_split


class BatchSampler:

    def __init__(self, data, batch_size, epochs=1):
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.indices = self._random_indices()

    def _permute(self):
        return array_split(permutation(len(self.data)), self.batch_size)

    def _random_indices(self):
        return deque(chain(*[self._permute() for _ in range(self.epochs)]))

    def sample(self):
        if len(self.indices) == 0:
            return None
        return self.data[self.indices.popleft()]