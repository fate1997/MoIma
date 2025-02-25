import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import \
    MolecularDescriptorCalculator

from moima.dataset._abc import FeaturizerABC
from moima.dataset.descriptor_vec.data import VecData


def _get_ecfp(mol: Chem.Mol, radius: int, n_bits: int) -> np.ndarray:
    r"""Get ECFP fingerprint."""
    morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol, 
        radius, 
        n_bits, 
        useChirality=True
    )
    morgan_fingerprint_array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(morgan_fingerprint, morgan_fingerprint_array)
    return morgan_fingerprint_array


def _get_rdkit_desc(mol: Chem.Mol) -> np.ndarray:
    r"""Get RDKit descriptors."""
    calc = MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    descriptors = calc.CalcDescriptors(mol)
    return np.array(descriptors)


def _get_dict_from_csv(csv_path: str) -> Dict[str, np.ndarray]:
    r"""Get descriptor dictionary from csv."""
    df = pd.read_csv(csv_path)
    assert 'smiles' in df.columns, 'smiles column not found in csv'
    desc_dict = {}
    for iter, row in df.iterrows():
        smiles = row['smiles']
        row_wo_smiles = row.drop('smiles')
        desc_dict[smiles] = np.array(row_wo_smiles.values).astype(np.float32)
    return desc_dict


def _get_desc_from_dict(key: str, desc_dict: Dict[str, Any]) -> np.ndarray:
    r"""Get descriptors from dictionary."""
    try:
        desc = desc_dict[key]
    except KeyError:
        warnings.warn(f'{key} not found in the dictionary, will be set to `None`')
        desc = None
    return desc


class DescFeaturizer(FeaturizerABC):
    r"""Featurizer for descriptors.
    
    Args:
        mol_desc (str): The descriptors to use. Available descriptors are 
            'ecfp', 'rdkit', 'csv', 'dict'. (default: :obj:`None`)
        ecfp_radius (int): The radius of ECFP. (default: :obj:`2`)
        ecfp_n_bits (int): The number of bits of ECFP. (default: :obj:`2048`)
        desc_csv_path (str): The path to the csv file containing descriptors. 
            (default: :obj:`None`)
        additional_desc_dict (dict): The dictionary containing additional 
            descriptors. Its key is SMILES and value is the descriptors.
            (default: :obj:`None`)
    """
    AVAILABLE_DESC = {
        'ecfp': _get_ecfp,
        'rdkit': _get_rdkit_desc,
        'csv': _get_dict_from_csv,
        'addi_dict': _get_desc_from_dict}
    
    def __init__(
        self, 
        mol_desc: str = None, 
        ecfp_radius: int = 2, 
        ecfp_n_bits: int = 2048,
        desc_csv_path: str = None,
        addi_desc_dict: Dict[str, torch.Tensor] = None
    ):
        if type(mol_desc) == str:
            desc_names = mol_desc.split(',')
        self.desc_csv_path = desc_csv_path
        self.desc_names = desc_names
        self.ecfp_radius = ecfp_radius
        self.ecfp_n_bits = ecfp_n_bits
        self.additional_desc_dict = addi_desc_dict
        
        if 'csv' in self.desc_names and self.desc_csv_path is None:
            raise ValueError('csv path is not provided.')
        if 'addi_dict' in self.desc_names and self.additional_desc_dict is None:
            raise ValueError('Additional descriptor dictionary is not provided.')
        
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
        if 'addi_dict' in self.desc_names:
            desc_length = len(list(self.additional_desc_dict.values())[0])
            self.columns.extend(f'desc_{i}' for i in range(desc_length))
    
    def __repr__(self) -> str:
        return f'DescFeaturizer(desc_names={self.desc_names}'
    
    def encode(self, mol: str) -> Optional[VecData]:
        r"""Encode a molecule to a vector data :obj:`VecData`."""
        rdmol = Chem.MolFromSmiles(mol)
        if rdmol is None:
            warnings.warn(f'Invalid smiles: {mol}')
            return None

        desc = []
        if 'ecfp' in self.desc_names:
            desc.append(_get_ecfp(rdmol, self.ecfp_radius, self.ecfp_n_bits))
        if 'rdkit' in self.desc_names:
            desc.append(_get_rdkit_desc(rdmol))
        if 'csv' in self.desc_names:
            desc_dict = _get_dict_from_csv(self.desc_csv_path)
            desc.append(_get_desc_from_dict(mol, desc_dict))
        if 'addi_dict' in self.desc_names:
            desc.append(_get_desc_from_dict(mol, self.additional_desc_dict))
            if _get_desc_from_dict(mol, self.additional_desc_dict) is None:
                return None
        desc = np.concatenate(desc)
        desc = torch.from_numpy(desc).float()
        return VecData(desc, None, mol)

    @property
    def arg4model(self) -> dict:
        return {'input_dim': len(self.columns)}