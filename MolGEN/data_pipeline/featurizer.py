from functools import partial
from typing import List

import torch
from tqdm import tqdm

from MolGEN.data_pipeline._util import one_hot_encoding
from MolGEN.data_pipeline._base import BaseFeaturizer

class SMILESFeaturizer(BaseFeaturizer):
    
    def __init__(self, 
                 charset: List[str], 
                 pad_length: int=120):
        self.pad_length = pad_length
        self.charset = charset
    
    def __dict__(self):
        dic = {
            'charset': self.charset,
            'pad_length': self.pad_length
        }
        return dic
    
    @classmethod
    def from_smiles_list(cls, 
                         smiles_list: List[str], 
                         pad_length: int=120) -> 'SMILESFeaturizer':
        r"""Create a SMILES featurizer from a list of SMILES strings.
        """
        charset = cls._get_charset(smiles_list)
        return cls(charset, pad_length)
    
    def _featurize(self, smiles: str) -> torch.Tensor:
        r"""Featurize a SMILES string into a one-hot encoding.
        """
        smiles = smiles.ljust(self.pad_length)
        one_hot_encoder = partial(one_hot_encoding, choices=self.charset)
        encodings = [one_hot_encoder(c) for c in smiles]
        return torch.tensor(encodings, dtype=torch.float32)
    
    @staticmethod
    def _get_charset(smiles_list: List) -> List[str]:
        r"""Get the charset of a list of SMILES strings.
        
        Args:
            smiles_list (list): A list of SMILES strings.
        
        Returns:
            A list of uniqe characters in the SMILES strings.
        """
        s = set()
        for smiles in smiles_list:
            for c in smiles:
                s.add(c)
        charset = [' '] + sorted(list(s))
        return charset
    
    def decode(self, x: torch.Tensor) -> List[str]:
        r"""Decode SMILES encodings into a SMILES list.
        
        Args:
            x (torch.Tensor): SMILES encoding, shape of [pad_length, 
                charset_length].
        
        Returns:
            A list of SMILES strings.
        """
        charset_idx = torch.argmax(x, dim=1)
        smiles = "".join(map(lambda x: self.charset[x], charset_idx)).strip()
        return smiles


if __name__ == '__main__':
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CCO',
        'CC(=O)O',
    ]
    featurizer = SMILESFeaturizer.from_smiles_list(smiles_list)
    x = featurizer(smiles_list[0])
    print(f"SMILES encodings shape: {x.shape}")
    smiles = featurizer.decode(x)
    print(f"SMILES list: {smiles}")