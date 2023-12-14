from abc import ABC, abstractmethod
from typing import Tuple

from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from moima.dataset._abc import DatasetABC


class SplitterABC(ABC):
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
    
    @abstractmethod
    def __call__(self, dataset: DatasetABC) \
                -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Split the dataset."""
    
    def create_loader(self, dataset: DatasetABC):
        if getattr(dataset, 'collate_fn', None):
            return DataLoader(dataset, 
                              batch_size=self.batch_size, 
                              shuffle=True,
                              collate_fn=dataset.collate_fn)
        else:
            return GeometricDataLoader(dataset, 
                                       batch_size=self.batch_size, 
                                       shuffle=True)