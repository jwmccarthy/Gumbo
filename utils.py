import numpy as np
import torch as th


def transpose_dict(d):
    return [dict(zip(d, v)) for v in zip(*d.values())]


def safe_from_numpy(v):
    return th.from_numpy(v) if isinstance(v, np.ndarray) else v