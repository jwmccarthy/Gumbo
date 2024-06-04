from typing import Dict

from torch import Tensor
from torch.utils.data import Dataset


class NamedTensorDataset(Dataset[Dict[str, Tensor]]):

    tensors: Dict[str, Tensor]

    def __init__(self, **tensors: Tensor) -> None:
        self.tensors = tensors

    def __getattr__(self, key):
        return self.tensors[key]

    def __getitem__(self, index):
        return NamedTensorDataset(**{k: v[index] for k, v in self.tensors.items()})
    
    def __setitem__(self, index, values):
        for i, k in enumerate(self.tensors): self.tensors[k][index] = values[i]
    
    def __len__(self):
        return next(iter(self.tensors)).size(0)
    
    def to(self, device):
        for v in self.tensors.values(): v.to(device)

    def cpu(self):
        for k, v in self.tensors.items(): self.tensors[k] = v.cpu()