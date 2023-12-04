from typing import List

import torch

from moima.dataset._abc import DataABC


class VecData(DataABC):
    """Data class for vector data."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor, smiles: str):
        super().__init__()
        
        self.x = x
        self.y = y
        self.smiles = smiles
    
    def __repr__(self) -> str:
        return f"VecData(shape of x: {self.x.shape}, smiles: {self.smiles})"
    
    def __len__(self):
        return len(self.smiles)


class VecBatch:
    """Batch class for vector data."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor, smiles: List[str]):
        self.x = x
        self.y = y
        self.smiles = smiles
            
    def __repr__(self) -> str:
        return f"SeqBatch(shape of x: {self.x.shape})"
    
    def __getitem__(self, index: int) -> VecData:
        return VecData(self.x[index], self.y[index], self.smiles[index])
    
    def to(self, device: torch.device) -> 'VecBatch':
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self