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
        self.seq_len = torch.tensor([len(l) for l in self.labels], dtype=torch.long)
    
    def __repr__(self) -> str:
        return f"SeqBatch(shape of x: {self.x.shape})"
    
    def __getitem__(self, index: int) -> SeqData:
        return SeqData(self.x[index], self.labels[index])
    
    def to(self, device: torch.device) -> 'SeqBatch':
        self.x = self.x.to(device)
        self.labels = self.seq_len.to(device)
        return self