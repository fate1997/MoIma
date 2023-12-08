from typing import List

import torch

from moima.dataset._abc import DataABC


class SeqData(DataABC):
    """Data class for sequence data."""
    def __init__(self, x: torch.Tensor, seq_len: torch.Tensor, smiles: str):
        super().__init__()
        
        self.x = x.long()
        assert self.x.shape == (x.size(0), )
        self.smiles = smiles
        self.seq_len = seq_len.long()
    
    def __repr__(self) -> str:
        return f"SeqData(shape of x: {self.x.shape}, smiles: {self.smiles})"
    
    def __len__(self):
        return len(self.smiles)


class SeqBatch:
    """Batch class for sequence data."""
    def __init__(self, x: torch.Tensor, seq_len: torch.Tensor, smiles: List[str]):
        self.x = x.long()
        assert self.x.size(0) == len(smiles)
        self.smiles = smiles
        self.seq_len = seq_len.long()
    
    def __repr__(self) -> str:
        return f"SeqBatch(shape of x: {self.x.shape})"
    
    def __getitem__(self, index: int) -> SeqData:
        seq_data = SeqData(self.x[index], self.seq_len[index], self.smiles[index])
        for k, v in self.__dict__.items():
            if k not in ['x', 'seq_len', 'smiles']:
                setattr(seq_data, k, v[index])
        return seq_data
    
    def to(self, device: torch.device) -> 'SeqBatch':
        self.x = self.x.to(device)
        self.seq_len = self.seq_len.to(device)
        return self