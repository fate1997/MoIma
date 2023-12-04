from typing import List, Union, Tuple

import pandas as pd
import torch

from .._abc import DatasetABC
from moima.dataset.descriptor_vec.featurizer import DescFeaturizer
from .data import VecBatch, VecData
from tqdm import tqdm


class DescDataset(DatasetABC):
    def __init__(self, 
                 raw_path: str, 
                 label_col: Union[str, List[str]],
                 featurizer: DescFeaturizer, 
                 processed_path: str = None, 
                 force_reload: bool = False, 
                 save_processed: bool = False):
        self.label_col = [label_col] if isinstance(label_col, str) else label_col
        super().__init__(raw_path, 
                         featurizer, 
                         processed_path, 
                         force_reload, 
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
        y = torch.stack(y, dim=0)
        return VecBatch(x, y, smiles)
    
    def _prepare_data(self):
        """Prepare data for the dataset."""
        assert self.raw_path.endswith('csv')
        df = pd.read_csv(self.raw_path)
        
        # Find the column containing SMILES
        smiles_col = self._get_smiles_column(df)

        # Featurize SMILES
        smiles_list = df[smiles_col].tolist()
        labels = df[self.label_col].to_numpy()
        print(f'labels.shape: {labels.shape}')
        if len(self.label_col) == 1:
            labels = labels.reshape(-1, 1)
        data_list = []
        for i, smiles in enumerate(tqdm(smiles_list, 'Descriptors generation')):
            data = self.featurizer(smiles, labels[i])
            if data is None:
                continue
            data_list.append(data)

        return data_list, self.featurizer.input_args