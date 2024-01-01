from typing import NamedTuple, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader

from moima.dataset._abc import DatasetABC
from moima.utils.splitter._abc import SplitterABC

IntOrFloat = Union[int, float]
Ratios = NamedTuple('Ratios', [('train', IntOrFloat), ('val', IntOrFloat)])


class RandomSplitter(SplitterABC):
    """Randomly split the dataset into train, val and test sets.
    
    Args:
        ratios (Ratios or tuple): The ratios of train and val sets. If the
            ratios are floats, they will be converted to integers according
            to the length of the dataset.
        split_test (bool): Whether to split the test set. Default to True.
        batch_size (int): The batch size of the dataloader. Default to 64.
        seed (int): The random seed. Default to 42.
    
    Examples:
        >>> splitter = RandomSplitter(ratios=(0.8, 0.1))
        >>> train_loader, val_loader, test_loader = splitter.split(dataset)
    """
    def __init__(self, 
                 frac_train: float = 0.8,
                 frac_val: float = 0.1, 
                 split_test: bool = True,
                 batch_size: int = 64,
                 seed: int = 42):
        super().__init__(frac_train, frac_val, split_test, batch_size)
        self.train_val = [frac_train, frac_val]
        self.split_test = split_test
        self.seed = seed
        self.batch_size = batch_size
        
    def _float2int(self, dataset_len: int):
        r"""Convert the float ratios to integers."""
        number_list = self.train_val
        if isinstance(number_list[0], float):
            number_list[0] = int(number_list[0] * dataset_len)
        if isinstance(number_list[1], float):
            number_list[1] = int(number_list[1] * dataset_len)
        self.train_val = number_list
        
    def __call__(self, dataset: DatasetABC) -> Tuple[DataLoader, DataLoader, DataLoader]:
        r"""Split the dataset into train, val and test data loaders.
        
        Args:
            dataset (DatasetABC): The dataset to be split.
        
        Returns:
            A tuple of train, val and test data loaders.
        """
        dataset.random_shuffle(self.seed)
        self._float2int(len(dataset))
        
        cum_sum = np.cumsum(self.train_val)
        
        train_dataset = dataset[:cum_sum[0]]
        val_dataset = dataset[cum_sum[0]:cum_sum[1]]
        
        train_loader = train_dataset.create_loader(self.batch_size, shuffle=True)
        val_loader = val_dataset.create_loader(self.batch_size, shuffle=False)
        
        if not self.split_test:
            return train_loader, val_loader, None
        
        test_dataset = dataset[cum_sum[1]:]
        test_loader = test_dataset.create_loader(self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader