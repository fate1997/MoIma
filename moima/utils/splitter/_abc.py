from abc import ABC, abstractmethod
from typing import Tuple
from moima.dataset._abc import DatasetABC
from torch.utils.data import DataLoader


class SplitterABC(ABC):
    
    @abstractmethod
    def __call__(self, dataset: DatasetABC) \
                -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Split the dataset."""