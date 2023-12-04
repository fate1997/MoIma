from typing import List

import pandas as pd
import torch
from tqdm import tqdm
import os

from moima.dataset._abc import DatasetABC, FeaturizerABC
from moima.dataset.smiles_seq.data import SeqBatch, SeqData
from moima.dataset.smiles_seq.featurizer import SeqFeaturizer


class SMILESSeq(DatasetABC):
    def __init__(self, 
                 raw_path: str,
                 featurizer: SeqFeaturizer,
                 vocab_path: str = None,
                 processed_path: str = None,
                 force_reload: bool = False,
                 save_processed: bool = False):
        self.vocab_path = vocab_path
        super().__init__(raw_path, featurizer, processed_path, force_reload, save_processed)
    
    @staticmethod
    def collate_fn(batch: List[SeqData]):
        """Collate function for the dataset."""
        x = torch.stack([b.x for b in batch])
        seq_len = torch.stack([b.seq_len for b in batch])
        smiles = [b.smiles for b in batch]
        return SeqBatch(x, seq_len, smiles)
        
    def _prepare_data(self):
        """Prepare data for the dataset."""
        assert self.raw_path.endswith('csv')
        df = pd.read_csv(self.raw_path)
        
        # Find the column containing SMILES
        smiles_col = self._get_smiles_column(df)
        
        # Featurize SMILES
        smiles_list = df[smiles_col].tolist()
        if self.vocab_path is not None:
            self.featurizer.load_vocab(self.vocab_path)
        else:
            self.featurizer.reload_charset(smiles_list)
            vocab_path = os.path.splitext(self.raw_path)[0] + '_vocab.pkl'
            self.featurizer.save_vocab(vocab_path)
        
        data_list = []
        for smiles in tqdm(smiles_list, 'SMILES featurization'):
            data = self.featurizer(smiles)
            data_list.append(data)

        return data_list, self.featurizer.input_args