import os
from typing import List

import pandas as pd
import torch

from moima.dataset._abc import DatasetABC
from moima.dataset.smiles_seq.data import SeqBatch, SeqData
from moima.dataset.smiles_seq.featurizer import SeqFeaturizer


class SeqDataset(DatasetABC):
    r"""The class for SMILES sequence dataset.
    
    Args:
        raw_path (str): The path to the raw data.
        featurizer_kwargs (dict): The keyword arguments for the featurizer.
        additional_cols (List[str]): The additional columns to be added to 
            :obj:`SeqData`, for example, the target values or the additional 
            features. (default: :obj:`None`)
        vocab_path (str): The path to the vocabulary file.
        processed_path (str): The path to the processed data.
        force_reload (bool): Whether to reload the data. (default: :obj:`False`)
        save_processed (bool): Whether to save the processed data. (default:
            :obj:`False`)
    """
    def __init__(self, 
                 raw_path: str,
                 featurizer_kwargs: dict,
                 additional_cols: List[str] = [],
                 vocab_path: str = None,
                 processed_path: str = None,
                 force_reload: bool = False,
                 save_processed: bool = False):
        self.vocab_path = vocab_path
        self.additional_cols = additional_cols
        super().__init__(raw_path, SeqFeaturizer, featurizer_kwargs,
                         processed_path, force_reload, save_processed)
    
    @staticmethod
    def collate_fn(batch: List[SeqData]) -> SeqBatch:
        """Collate function for the dataset."""
        x = torch.stack([b.x for b in batch])
        seq_len = torch.stack([b.seq_len for b in batch])
        smiles = [b.smiles for b in batch]
        result_batch = SeqBatch(x, seq_len, smiles)
        for key in batch[0].__dict__:
            if key not in ['x', 'seq_len', 'smiles']:
                setattr(result_batch, 
                        key, 
                        torch.stack([getattr(b, key) for b in batch]))
        return result_batch
    
    def prepare(self):
        r"""Prepare data for the dataset. Only support csv files.
        """
        assert self.raw_path.endswith('csv')
        df = pd.read_csv(self.raw_path)
        
        # Find the column containing SMILES
        smiles_col = self._get_smiles_column(df)
        
        # Load vocabulary
        smiles_list = df[smiles_col].tolist()
        if self.vocab_path is not None:
            self.featurizer.load_vocab(self.vocab_path)
        else:
            self.featurizer.reload_vocab(smiles_list)
            vocab_path = os.path.splitext(self.raw_path)[0] + '_vocab.pkl'
            self.featurizer.save_vocab(vocab_path)
        
        # Extract additional columns
        print(self.additional_cols)
        additional_kwargs = dict(zip(self.additional_cols,
                                     df[self.additional_cols].values.T))

        data_list = self.featurizer(smiles_list, **additional_kwargs)

        return data_list