import numpy as np
import torch as th


def transpose_dict(d):
    return [dict(zip(d, v)) for v in zip(*d.values())]


def try_from_numpy(v):
    return th.from_numpy(v) if isinstance(v, np.ndarray) else v


def array_split(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]