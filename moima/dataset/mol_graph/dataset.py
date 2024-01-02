import os
from typing import List, Union

import pandas as pd
import torch

from moima.dataset._abc import DatasetABC
from moima.dataset.mol_graph.data import GraphData
from torch_geometric.data import Batch as GraphBatch
from moima.dataset.mol_graph.featurizer import GraphFeaturizer
from rdkit import Chem


class GraphDataset(DatasetABC):
    def __init__(self, 
                 raw_path: str,
                 label_path: str = None,
                 remove_hydrogen: bool = False,
                 label_col: Union[str, List[str]] = None,
                 additional_cols: List[str] = [],
                 featurizer: GraphFeaturizer = None,
                 processed_path: str = None,
                 force_reload: bool = False,
                 save_processed: bool = False):
        self.label_col = [label_col] if isinstance(label_col, str) else label_col
        self.label_path = label_path
        self.remove_hydrogen = remove_hydrogen
        self.additional_cols = additional_cols
        super().__init__(raw_path, featurizer, processed_path, 
                         force_reload, save_processed)
    
    def prepare(self) -> List[GraphData]:
        r"""Prepare data for the dataset."""
        if self.raw_path.endswith('.csv'):
            df = pd.read_csv(self.raw_path)
            # Find the column containing SMILES
            smiles_col = self._get_smiles_column(df)
            mols = df[smiles_col].tolist()
            self.label_path = self.raw_path
        
        elif self.raw_path.endswith('.sdf'):
            mols = Chem.SDMolSupplier(self.raw_path, 
                                       removeHs=self.remove_hydrogen, 
                                       sanitize=False)
        
        if self.label_path is not None:
            assert self.label_path.endswith('.csv')
            df = pd.read_csv(self.label_path)
            assert len(df) == len(mols)
    
        # Get labels and additional columns
        labels = df[self.label_col].to_numpy()
        if len(self.label_col) == 1:
            labels = labels.reshape(-1, 1)
        additional_kwargs = dict(zip(self.additional_cols,
                                    df[self.additional_cols].values.T))
        additional_kwargs.update({'y': labels})
        data_list = self.featurizer(mols, **additional_kwargs)        
        return data_list