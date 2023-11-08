from abc import ABC
from typing import Any

from torch.utils.data import Dataset
import torch
import os
from typing import Union


class BaseData(ABC):
    pass


class BaseFeaturizer(ABC):
    
    def __call__(self, *args: Any, **kwds: Any) -> Union[torch.Tensor, BaseData]:
        return self._featurize(*args, **kwds)
    
    def _featurize(self, *args, **kwargs):
        raise NotImplementedError
    
    def __dict__(self):
        raise NotImplementedError
    
    @classmethod
    def from_dict(cls, **kwargs):
        return cls(**kwargs)


class BaseDataset(Dataset, ABC):
    
    def __init__(self,
                 raw_path: str,
                 Featurizer: BaseFeaturizer,
                 featurizer_config: dict=None,
                 processed_path: str = None,
                 replace: bool = False):
        super().__init__()
        
        self.raw_path = raw_path
        self.featurizer_config = featurizer_config
        self.replace = replace
        
        if processed_path is None:
            processed_path = os.path.splitext(raw_path)[0] + '.pt'
        
        if os.path.exists(processed_path) and not replace:
            processed_dataset = torch.load(processed_path)
            self.data_list = processed_dataset['data_list']
            featurizer_dict = processed_dataset['featurizer']
            self.featurizer = self.load_featurizer(Featurizer, featurizer_dict)
        else:
            self.data_list, featurizer_dict = self._prepare_data()
            self.featurizer = self.load_featurizer(Featurizer, featurizer_dict)
            torch.save({'data_list': self.data_list,  
                        'featurizer': featurizer_dict}, processed_path)
    
    @staticmethod
    def load_featurizer(Featurizer: BaseFeaturizer, 
                        featurizer_dict: dict):
        return Featurizer.from_dict(**featurizer_dict)
    
    def _prepare_data(self):
        r"""Prepare data for the dataset."""
        raise NotImplementedError
    
    def __getitem__(self, index):
        return self.data_list[index]
    
    def __len__(self):
        return len(self.data_list)