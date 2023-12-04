import warnings
from typing import Any, List, Optional, Dict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import \
    MolecularDescriptorCalculator

from .._abc import DataABC, FeaturizerABC
from .data import VecData
import torch


def _get_ecfp(mol: Chem.Mol, radius: int, n_bits: int) -> np.ndarray:
    morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 
                                                            radius, 
                                                            n_bits, 
                                                            useChirality=True)
    morgan_fingerprint_array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(morgan_fingerprint, morgan_fingerprint_array)
    return morgan_fingerprint_array


def _get_rdkit_desc(mol: Chem.Mol):
    calc = MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    descriptors = calc.CalcDescriptors(mol)
    return np.array(descriptors)


def _get_desc_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    assert 'smiles' in df.columns, 'smiles column not found in csv'
    desc_dict = {}
    for row in df.iterrows():
        smiles = row['smiles']
        desc_dict[smiles] = row.drop('smiles').values.numpy()
    return desc_dict


class DescFeaturizer(FeaturizerABC):
    AVAILABLE_DESC = {
        'ecfp': _get_ecfp,
        'rdkit': _get_rdkit_desc,
        'csv': _get_desc_from_csv,
        'dict': None}
    def __init__(self, 
                mol_desc: str = None, 
                ecfp_radius: int = 2, 
                ecfp_n_bits: int = 2048,
                desc_csv_path: str = None,
                additional_desc_dict: Dict[str, torch.Tensor] = None):
        if type(mol_desc) == str:
            desc_names = mol_desc.split(',')
        self.desc_csv_path = desc_csv_path
        self.desc_names = desc_names
        self.ecfp_radius = ecfp_radius
        self.ecfp_n_bits = ecfp_n_bits
        self.additional_desc_dict = additional_desc_dict
        
        for name in self.desc_names:
            if name not in self.AVAILABLE_DESC:
                raise ValueError(f'{name} not found in {list(self.AVAILABLE_DESC.keys())}')
        
        self.columns = []
        if 'ecfp' in self.desc_names:
            self.columns.extend([f'ecfp_{i}' for i in range(self.ecfp_n_bits)])
        if 'rdkit' in self.desc_names:
            self.columns.extend([x[0] for x in Descriptors._descList])
        if 'csv' in self.desc_names:
            self.columns.extend(pd.read_csv(self.desc_csv_path).columns[1:])
        if 'dict' in self.desc_names:
            self.columns.extend(f'desc_{i}' for i in range(len(self.additional_desc_dict)))
    
    def __repr__(self) -> str:
        return f'DescFeaturizer(desc_names={self.desc_names}'
    
    def __call__(self, smiles: str, labels: Any = None) -> Optional[VecData]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f'Invalid smiles: {smiles}')
            return None

        desc = []
        if 'ecfp' in self.desc_names:
            desc.append(_get_ecfp(mol, self.ecfp_radius, self.ecfp_n_bits))
        if 'rdkit' in self.desc_names:
            desc.append(_get_rdkit_desc(mol))
        if 'csv' in self.desc_names:
            desc_dict = _get_desc_from_csv(self.desc_csv_path)
            desc.append(desc_dict[smiles])
        if 'dict' in self.desc_names:
            desc.append(self.additional_desc_dict[smiles])
        desc = np.concatenate(desc)
        
        desc = torch.from_numpy(desc).float()
        if labels is not None:
            labels = torch.tensor(labels).float()
        return VecData(desc, labels, smiles)

    @property
    def input_args(self):
        dic = {
            'desc_names': self.desc_names,
            'ecfp_radius': self.ecfp_radius,
            'ecfp_n_bits': self.ecfp_n_bits,
            'desc_csv_path': self.desc_csv_path
        }
        return dic