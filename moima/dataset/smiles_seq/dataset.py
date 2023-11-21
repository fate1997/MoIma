from typing import List

import pandas as pd
import torch
from tqdm import tqdm

from moima.dataset._abc import DatasetABC, FeaturizerABC
from moima.dataset.smiles_seq.data import SeqData, SeqBatch
from moima.dataset.smiles_seq.featurizer import SeqFeaturizer


class SMILESSeq(DatasetABC):
    def __init__(self, 
                 raw_path: str,
                 featurizer_config: dict,
                 Featurizer: FeaturizerABC = SeqFeaturizer,
                 processed_path: str = None,
                 force_reload: bool = False,
                 save_processed: bool = False):
        super().__init__(raw_path, featurizer_config, Featurizer,
                         processed_path, force_reload, save_processed)
    
    @staticmethod
    def collate_fn(batch: List[SeqData]):
        """Collate function for the dataset."""
        x = torch.stack([b.x for b in batch])
        labels = [b.label for b in batch]
        return SeqBatch(x, labels)
        
    def _prepare_data(self):
        """Prepare data for the dataset."""
        assert self.raw_path.endswith('csv')
        df = pd.read_csv(self.raw_path)
        
        # Find the column containing SMILES
        smiles_col = None
        for column in [c.lower() for c in df.columns]:
            if 'smiles' in column and smiles_col is None:
                smiles_col = column
                continue
            if 'smiles' in column and smiles_col is not None:
                raise ValueError('Multiple columns contain "smiles"')
        if smiles_col is None:
            raise ValueError('No column contains "smiles"')

        # Featurize SMILES
        smiles_list = df[smiles_col].tolist()
        self.featurizer.reload_charset(smiles_list)
        
        data_list = []
        for smiles in tqdm(smiles_list, 'SMILES featurization'):
            data = self.featurizer(smiles)
            data_list.append(data)

        return data_list, self.featurizer.input_args