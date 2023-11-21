from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from moima.dataset._abc import DatasetABC
from moima.trainer.splitter._abc import SplitterABC

IntOrFloat = Union[int, float]
Ratios = NamedTuple('Ratios', [('train', IntOrFloat), 
                               ('val', IntOrFloat)])


class RandomSplitter(SplitterABC):
    def __init__(self, 
                 ratios: Union[Ratios, tuple] = Ratios(train=0.8, val=0.1), 
                 split_test: bool = True,
                 batch_size: int = 64,
                 seed: int = 42):
        
        assert len(ratios) == 2
        if isinstance(ratios, tuple):
            ratios = Ratios(*ratios)
        
        self.split_test = split_test
        self.ratios = ratios
        self.seed = seed
        self.batch_size = batch_size
        
    def _float2int(self, dataset_len: int):
        number_list = [self.ratios.train, self.ratios.val]
        if isinstance(self.ratios.train, float):
            number_list[0] = int(self.ratios.train * dataset_len)
        if isinstance(self.ratios.val, float):
            number_list[1] = int(self.ratios.val * dataset_len)
        self.ratios = Ratios(*number_list)
    
    def _create_loader(self, dataset: DatasetABC):
        if getattr(dataset, 'collate_fn', None):
            return DataLoader(dataset, 
                              batch_size=self.batch_size, 
                              shuffle=True,
                              collate_fn=dataset.collate_fn)
        else:
            return GeometricDataLoader(dataset, 
                                       batch_size=self.batch_size, 
                                       shuffle=True)
    
    def split(self, dataset: DatasetABC) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset.random_shuffle(self.seed)
        self._float2int(len(dataset))
        
        cum_sum = np.cumsum(list(self.ratios))
        
        train_dataset = dataset[:cum_sum[0]]
        val_dataset = dataset[cum_sum[0]:cum_sum[1]]
        
        train_loader = self._create_loader(train_dataset)
        val_loader = self._create_loader(val_dataset)
        
        if not self.split_test:
            return train_loader, val_loader, None
        
        test_dataset = dataset[cum_sum[1]:]
        test_loader = self._create_loader(test_dataset)
        return train_loader, val_loader, test_loader