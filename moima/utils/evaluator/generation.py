import json
import os
from typing import List, Literal, Optional

import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

from moima.pipeline.pipe import PipeABC
from moima.typing import MolRepr

RDLogger.DisableLog('rdApp.*')


class GenerationMetrics:
    r"""Metrics for evaluating the generation performance of a model.
    
    Args:
    """
    AVAIL_METRICS = ['valid', 'unique', 'novel', 'recon_accuracy']
    def __init__(self, sampled_mols: MolRepr, train_mols: MolRepr):
        self.sampled_smiles = list(map(self.mol2smiles, sampled_mols))
        self.sampled_rdmols = list(map(self.smiles2mol, sampled_mols))
        self.train_smiles = list(map(self.mol2smiles, train_mols))
        self.train_rdmols = list(map(self.smiles2mol, train_mols))
    
    @staticmethod
    def mol2smiles(mol: MolRepr) -> str:
        if isinstance(mol, str):
            return mol
        elif isinstance(mol, Chem.Mol):
            return Chem.MolToSmiles(mol)
        else:
            raise TypeError(f'Unsupported type {type(mol)}')
    
    @staticmethod
    def smiles2mol(mol: MolRepr) -> Optional[Chem.Mol]:
        if isinstance(mol, str):
            return Chem.MolFromSmiles(mol)
        elif isinstance(mol, Chem.Mol):
            return mol
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
    
    def to_json(self, metric_names: List[str] = None):
        metrics = self.get_metrics(metric_names)
        file_path = os.path.join(self.pipe.config.output_folder,
                                 f'generation_metrics_{self.pipe.config.desc}.json')
        with open(file_path, 'w') as f:
            json.dump(metrics, f)
    
    def rand_sample(self, num_samples: int) -> List[Optional[Chem.Mol]]:
        if getattr(self, 'valid_mols', None) is not None:
            return self.valid_mols
        sampled_smiles = self.pipe.sample(num_samples)
        mols = []
        for smiles in tqdm(sampled_smiles):
            mol = Chem.MolFromSmiles(smiles)
            mols.append(mol)
        valid_mols = [mol is not None for mol in mols]
        self.valid_mols = [mols[i] for i, valid in enumerate(valid_mols) if valid]
        return self.valid_mols
    
    @property
    def valid(self):
        valid_mols = self.rand_sample(self.num_samples)
        return len(valid_mols) / self.num_samples
    
    @property
    def unique(self):
        valid_mols = self.rand_sample(self.num_samples)
        unique_mols = set([Chem.MolToSmiles(mol) for mol in valid_mols])
        return len(unique_mols) / len(valid_mols)
    
    @property
    def novel(self):
        valid_mols = self.rand_sample(self.num_samples)
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        train_smiles = []
        for batch in self.pipe.train_loader:
            for smiles in batch.smiles:
                train_smiles.append(smiles)
        return len(set(valid_smiles) - set(train_smiles)) / len(valid_smiles)
    
    @property
    def recon_accuracy(self):
        count = 0
        success = 0
        for batch in tqdm(self.loader, desc='Model Running'):
            self.pipe.model.to(self.device)
            x_hat, _ = self.pipe._forward_batch(batch)
            x_hat = x_hat.detach().cpu()
            
            for i, x in enumerate(x_hat):
                recon_output = self.pipe.featurizer.decode(x, is_raw=False)
                original = batch.smiles[i]
                count += 1
                if recon_output == original:
                    success += 1
        return success / count