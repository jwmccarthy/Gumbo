import numpy as np
from collections import deque


def batch_sample(data, batch_size):
    random_inds = np.random.permutation(len(data))

    for batch_start in range(0, len(data), batch_size):
        batch_end = batch_start + batch_size
        batch_inds = random_inds[batch_start:batch_end]
        yield data[batch_inds]


class BatchSampler:

    def __init__(self, data):
        self.data = data
        self.indices = self._random_indices()

    def _random_indices(self):
        return np.random.permutation(len(self.data))

    def sample(self, batch_size):
        if len(self.indices) == 0:
            self.indices = self._random_indices()
        batch = self.data[self.indices[:batch_size]]
        self.indices = self.indices[batch_size:]
        return batch