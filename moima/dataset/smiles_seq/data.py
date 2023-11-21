from typing import List

import torch
from moima.dataset._abc import DataABC


class SeqData(DataABC):
    def __init__(self, x: torch.Tensor, label: str):
        super().__init__()
        
        self.x = x.long()
        assert self.x.shape == (x.size(0), )
        self.label = label
    
    def __repr__(self) -> str:
        return f"SeqData(shape of x: {self.x.shape}, labels: {self.label})"
    
    def __len__(self):
        return len(self.label)


class SeqBatch:
    def __init__(self, x: torch.Tensor, labels: List[str]):
        self.x = x.long()
        assert self.x.size(0) == len(labels)
        self.labels = labels
    
    def __repr__(self) -> str:
        return f"SeqBatch(shape of x: {self.x.shape})"
    
    @property
    def seq_len(self):
        return torch.tensor([len(l) for l in self.labels], dtype=torch.long)