import numpy as np


def batch_sample(data, batch_size):
    random_inds = np.random.permutation(len(data))

    for batch_start in range(0, len(data), batch_size):
        batch_end = batch_start + batch_size
        batch_inds = random_inds[batch_start:batch_end]
        yield data[batch_inds]
