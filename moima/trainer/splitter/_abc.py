from abc import ABC, abstractmethod
from moima.dataset._abc import DatasetABC


class SplitterABC(ABC):
    
    @abstractmethod
    def split(self, dataset: DatasetABC):
        """Split the dataset."""