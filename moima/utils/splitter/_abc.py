from abc import ABC, abstractmethod
from typing import Tuple

from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from moima.dataset._abc import DatasetABC


class SplitterABC(ABC):
    
    def __init__(self, 
                 frac_train: float,
                 frac_val: float, 
                 split_test: bool,
                 batch_size: int):
        self.frac_train = frac_train
        self.frac_val = frac_val
        self.split_test = split_test
        self.batch_size = batch_size
    
    @abstractmethod
    def __call__(self, dataset: DatasetABC) \
                -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Split the dataset."""