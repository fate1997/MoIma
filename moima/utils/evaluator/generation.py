import json
import os
from typing import List, Literal, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from rdkit import Chem, RDLogger
from tqdm import tqdm

from moima.pipeline.pipe import PipeABC
from moima.typing import MolRepr
from moima.utils.evaluator.sascorer import calculateScore

RDLogger.DisableLog('rdApp.*')


class GenerationMetrics:
    r"""Metrics for evaluating the generation performance of a model.
    
    Args:
    """
    AVAIL_METRICS = ['valid', 'unique', 'novel', 'recon_accuracy', 'sascore']
    def __init__(self, 
                 sampled_mols: List[MolRepr], 
                 train_mols: List[MolRepr],
                 input_mols: List[MolRepr]=None,
                 recon_mols: List[MolRepr]=None):
        self.sampled_smiles = list(map(self.mol2smiles, sampled_mols))
        self.sampled_rdmols = list(map(self.smiles2mol, sampled_mols))
        self.num_samples = len(self.sampled_smiles)
        self.train_smiles = list(map(self.mol2smiles, train_mols))
        self.train_rdmols = list(map(self.smiles2mol, train_mols))
       
        if input_mols is not None:
            self.input_smiles = list(map(self.mol2smiles, input_mols))
        if recon_mols is not None:
            self.recon_smiles = list(map(self.mol2smiles, recon_mols))
        
        self.valid_mols = [mol for mol in self.sampled_rdmols if mol is not None]
    
    @staticmethod
    def mol2smiles(mol: MolRepr) -> str:
        if isinstance(mol, str):
            return mol
        elif isinstance(mol, Chem.Mol):
            return Chem.MolToSmiles(mol)
        elif mol is None:
            return None
        else:
            raise TypeError(f'Unsupported type {type(mol)}')
    
    @staticmethod
    def smiles2mol(mol: MolRepr) -> Optional[Chem.Mol]:
        if isinstance(mol, str):
            return Chem.MolFromSmiles(mol)
        elif isinstance(mol, Chem.Mol):
            return mol
        elif mol is None:
            return None
        else:
            raise TypeError(f'Unsupported type {type(mol)}')
    
    def get_metrics(self, metric_names: List[str] = None, save_path: str = None):
        if metric_names is None:
            metric_names = self.AVAIL_METRICS
        
        metrics = {}
        for metric in metric_names:
            metrics[metric] = getattr(self, metric)
        
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(metrics, f)
        return metrics
            
    @property
    def valid(self):
        return len(self.valid_mols) / self.num_samples
    
    @property
    def sascore(self):
        score = 0
        for mol in self.valid_mols:
            score += calculateScore(mol)
        return score / len(self.valid_mols)
    
    @property
    def unique(self):
        unique_mols = set([Chem.MolToSmiles(mol) for mol in self.valid_mols])
        return len(unique_mols) / len(self.valid_mols)
    
    @property
    def novel(self):
        valid_smiles = [Chem.MolToSmiles(mol) for mol in self.valid_mols]
        train_smiles = self.train_smiles
        return len(set(valid_smiles) - set(train_smiles)) / len(valid_smiles)
    
    @property
    def recon_accuracy(self):
        success = 0
        for i in range(len(self.input_smiles)):
            if self.input_smiles[i] == self.recon_smiles[i]:
                success += 1
        return success / len(self.input_smiles)