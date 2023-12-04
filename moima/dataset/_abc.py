import os
from abc import ABC, abstractmethod
from typing import Any, Union, List

import torch
from torch_geometric.data import Dataset
import numpy as np
import pandas as pd


IndexType = Union[slice, torch.Tensor, np.ndarray, List[int]]


class DataABC(ABC):
    """Data abstract base class."""
    
    @abstractmethod
    def __len__(self):
        """Return the length of the data."""
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return the representation of the data."""


class FeaturizerABC(ABC):
    """Featurizer abstract base class."""
    
    @abstractmethod
    def __call__(self, smiles: str, labels: Any = None) -> DataABC:
        """Featurize the input raw data to `BaseData`."""
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return the representation of the featurizer."""
    
    @property
    @abstractmethod
    def input_args(self):
        """Return the input arguments of the featurizer."""
    
    @classmethod
    def from_dict(cls, **kwargs) -> 'FeaturizerABC':
        """Create a featurizer from a dictionary."""
        return cls(**kwargs)


class DatasetABC(Dataset):
    """Dataset abstract base class."""
    
    def __init__(self,
                 raw_path: str,
                 featurizer: FeaturizerABC,
                 processed_path: str = None,
                 force_reload: bool = False,
                 save_processed: bool = False,
                 **kwargs):
        super().__init__()
        
        self.raw_path = raw_path
        self.force_reload = force_reload
        self.save_processed = save_processed
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.featurizer = featurizer
        
        if processed_path is None:
            processed_path = os.path.splitext(raw_path)[0] + '.pt'
        
        if os.path.exists(processed_path) and not force_reload:
            processed_dataset = torch.load(processed_path)
            self.data_list = processed_dataset['data_list']
            featurizer_dict = processed_dataset['featurizer']
            
        else:
            self.data_list, featurizer_dict = self._prepare_data()

        if save_processed:
            torch.save({'data_list': self.data_list,  
                        'featurizer': featurizer_dict}, processed_path)
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(Number of data: {len(self)})'
    
    @abstractmethod
    def _prepare_data(self):
        """Prepare data for the dataset."""
        
    def get(self, idx: int) -> DataABC:
        """Gets the data object at index `idx`."""
        return self.data_list[idx]
    
    def _get_smiles_column(self, df: pd.DataFrame) -> str:
        """Return the column containing SMILES."""
        smiles_col = None
        lower_columns = [c.lower() for c in df.columns]
        for i, column in enumerate(lower_columns):
            if 'smiles' in column and smiles_col is None:
                smiles_col = i
                continue
            if 'smiles' in column and smiles_col is not None:
                raise ValueError('Multiple columns contain "smiles"')
        if smiles_col is None:
            raise ValueError('No column contains "smiles"')
        smiles_col = df.columns[smiles_col]
        return smiles_col
    
    def len(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_list)
    
    def random_shuffle(self, seed: int = 42):
        """Randomly shuffle the dataset."""
        np.random.seed(seed)
        np.random.shuffle(self.data_list)