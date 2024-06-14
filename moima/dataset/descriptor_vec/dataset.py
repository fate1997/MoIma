from typing import List, Union

import pandas as pd
import torch

from moima.dataset._abc import DatasetABC
from moima.dataset.descriptor_vec.data import VecBatch, VecData
from moima.dataset.descriptor_vec.featurizer import DescFeaturizer


class DescDataset(DatasetABC):
    def __init__(self, 
                 raw_path: str, 
                 featurizer: DescFeaturizer,
                 label_col: Union[str, List[str]],
                 additional_cols: List[str] = [],
                 processed_path: str = None, 
                 force_reload: bool = False, 
                 save_processed: bool = False):
        self.label_col = [label_col] if isinstance(label_col, str) else label_col
        self.additional_cols = additional_cols
        super().__init__(raw_path, featurizer, processed_path, force_reload, 
                         save_processed)
        
    @staticmethod
    def collate_fn(batch: List[VecData]):
        """Collate function for the dataset."""
        x, y, smiles = [], [], []
        for data in batch:
            x.append(data.x)
            y.append(data.y)
            smiles.append(data.smiles)
        x = torch.stack(x, dim=0)
        y = torch.cat(y, dim=0)
        
        result_batch = VecBatch(x, y, smiles)
        for key in batch[0].__dict__:
            if key not in ['x', 'y', 'smiles']:
                setattr(result_batch, 
                        key, 
                        torch.stack([getattr(b, key) for b in batch]))
        return result_batch
    
    def prepare(self):
        """Prepare data for the dataset."""
        assert self.raw_path.endswith('csv')
        df = pd.read_csv(self.raw_path)
        
        # Find the column containing SMILES
        smiles_col = self._get_smiles_column(df)

        # Get SMILES, labels and additional columns
        smiles_list = df[smiles_col].tolist()
        labels = df[self.label_col].to_numpy()
        if len(self.label_col) == 1:
            labels = labels.reshape(-1, 1)
        additional_kwargs = dict(zip(self.additional_cols,
                                     df[self.additional_cols].values.T))
        additional_kwargs.update({'y': torch.FloatTensor(labels)})
        
        data_list = self.featurizer(smiles_list, **additional_kwargs)

        return data_list