import os
from abc import ABC, abstractmethod, abstractproperty
from typing import List, Callable, Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

from moima.typing import IndexType, MolRepr


class DataABC(ABC):
    r"""Data abstract class."""
    
    @abstractmethod
    def __len__(self):
        """Return the length of the data."""
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return the representation of the data."""


class FeaturizerABC(ABC):
    r"""Featurizer abstract class."""
        
    @abstractmethod
    def __repr__(self) -> str:
        """Return the representation of the featurizer."""
    
    def __call__(self, 
                 mol_list: List[MolRepr],
                 **kwargs) -> List[DataABC]:
        """Featurize a batch of SMILES.
        
        Args:
            mol_list: A list of SMILES.
            **kwargs: Other features. The key is the name of the feature, and 
                the value is a list of the feature. The length of the list should
                be the same as that of `mol_list`.
        
        Returns:
            A list of Data.
        """
        data_list = []
        for i, mol in enumerate(tqdm(mol_list, 'Featurization')):
            data = self.encode(mol)
            if data is None:
                continue            
            for key, value in kwargs.items():
                assign_value = value[i]
                if not isinstance(value[i], str):
                    assign_value = torch.as_tensor(value[i])
                    if assign_value.ndim == 1:
                        assign_value = assign_value.unsqueeze(-1)
                setattr(data, key, assign_value)
            data_list.append(data)
        return data_list        
    
    @abstractmethod
    def encode(self, mol: MolRepr) -> DataABC:
        """Featurize the input raw data to Data."""
    
    @abstractproperty
    def arg4model(self) -> dict:
        """Return the arguments for the model."""

class DatasetABC(Dataset):
    r"""Dataset abstract base class.
    
    Args:
        raw_path (str): The path to the raw data.
        featurizer_cls (FeaturizerABC): The featurizer class.
        featurizer_kwargs (dict): The keyword arguments for the featurizer. 
            (default: :obj:`None`)
        processed_path (str): The path to the processed data. If None, the
            processed data will be saved to the same directory as `raw_path`, 
            and end with '.pt'. (default: :obj:`None`)
        force_reload (bool): Whether to force reload the processed data. (default:
            :obj:`False`)
        save_processed (bool): Whether to save the processed data. (default:
            :obj:`False`)
    """
    
    def __init__(
        self,
        raw_path: str,
        featurizer: FeaturizerABC,
        processed_path: str=None,
        force_reload: bool=False,
        save_processed: bool=False
    ):
        super().__init__()
        self.raw_path = raw_path
        self.featurizer = featurizer
        self.force_reload = force_reload
        self.save_processed = save_processed
        self.processed_path = processed_path
        
        if not processed_path:
            processed_path = os.path.splitext(raw_path)[0] + '.pt'
        
        if os.path.exists(processed_path) and not force_reload:
            load_result = torch.load(processed_path)
            self.data_list = load_result['data_list']
            self.featurizer = load_result['featurizer']
        else:
            self.data_list = self.prepare()
            if save_processed:
                self.save(processed_path)
    
    def __repr__(self) -> str:
        r"""Return the representation of the dataset."""
        return self.__class__.__name__ + f'(Number of data: {len(self)})'
    
    @abstractmethod
    def prepare(self):
        r"""Prepare data for the dataset."""
        
    def get(self, idx: int) -> DataABC:
        r"""Gets the data object at index `idx`."""
        return self.data_list[idx]
    
    def _get_smiles_column(self, df: pd.DataFrame) -> str:
        r"""Return the column containing SMILES."""
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
    
    def filter(self, drop_ids: List[int]):
        filtered_data_list = [data for i, data in enumerate(self.data_list) if i not in drop_ids]
        self.data_list = filtered_data_list
    
    def apply_transform(self, func: Callable[[DataABC, Any], DataABC], **kwargs):
        for i, data in enumerate(self.data_list):
            self.data_list[i] = func(data, **kwargs)
    
    def save(self, path: str=None):
        if path is None:
            path = self.processed_path
        torch.save({'data_list': self.data_list,
                    'featurizer': self.featurizer}, path)    
    
    def len(self) -> int:
        r"""Return the length of the dataset."""
        return len(self.data_list)
    
    def random_shuffle(self, seed: int = 42):
        r"""Randomly shuffle the dataset."""
        np.random.seed(seed)
        np.random.shuffle(self.data_list)
    
    def create_loader(self, batch_size: int, shuffle: bool=True) -> DataLoader:
        r"""Create a data loader for the dataset."""
        if getattr(self, 'collate_fn', None):
            return DataLoader(self, 
                              batch_size=batch_size, 
                              shuffle=shuffle,
                              collate_fn=self.collate_fn)
        else:
            return GeometricDataLoader(self, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle)