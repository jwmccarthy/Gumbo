from typing import Dict, Sequence

from torch import Tensor
from torch.utils.data import Dataset


class NamedTensorDataset(Dataset[Dict[str, Tensor]]):

    tensors: Dict[str, Tensor]

    def __init__(self, **tensors: Tensor):
        self.tensors = tensors

    def __getattr__(self, key):
        return self.tensors[key]

    def __getitem__(self, index):
        return NamedTensorDataset(**{k: v[index] for k, v in self.tensors.items()})
    
    def __setitem__(self, index, values):
        for i, k in enumerate(self.tensors): self.tensors[k][index] = values[i]
    
    def __len__(self):
        return next(iter(self.tensors.values())).size(0)
    
    def to(self, device):
        for v in self.tensors.values(): v.to(device)

    def cpu(self):
        for k, v in self.tensors.items(): self.tensors[k] = v.cpu()


class Subset(Dataset):

    dataset: Dataset
    indices: Tensor

    def __init__(self, dataset: Dataset, indices: slice, **kwargs):
        self.dataset = dataset
        self.indices = indices
        self.__dict__.update(kwargs)  # auxiliary attributes i.e. final obs

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self):
        return self.indices.stop - self.indices.start + 1