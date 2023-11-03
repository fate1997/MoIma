from functools import partial
from typing import List

import torch
from tqdm import tqdm

from MolGEN.dataset._util import one_hot_encoding


class SMILESFeaturizer:
    
    def __init__(self, 
                 charset: List[str], 
                 pad_length: int=120):
        self.pad_length = pad_length
        self.charset = charset
    
    @classmethod
    def from_smiles_list(cls, 
                         smiles_list: List[str], 
                         pad_length: int=120) -> 'SMILESFeaturizer':
        r"""Create a SMILES featurizer from a list of SMILES strings.
        """
        charset = cls._get_charset(smiles_list)
        return cls(charset, pad_length)
    
    def __call__(self, smiles_list: List[str]):
        r"""Featurize a list of SMILES strings into a tensor.
        """
        embeddings = []
        for smiles in tqdm(smiles_list, desc='SMILES featurization'):
            embeddings.append(self._featurize(smiles))
        return torch.stack(embeddings)
    
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
            x (torch.Tensor): SMILES encodings, shape of [num_smiles, pad_length, 
                charset_length].
        
        Returns:
            A list of SMILES strings.
        """
        smiles_list = []
        for encoding in x:
            charset_idx = torch.argmax(encoding, dim=1)
            smiles = "".join(map(lambda x: self.charset[x], charset_idx)).strip()
            smiles_list.append(smiles)
        return smiles_list


if __name__ == '__main__':
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CCO',
        'CC(=O)O',
    ]
    featurnizer = SMILESFeaturizer.from_smiles_list(smiles_list)
    x = featurnizer(smiles_list)
    print(f"SMILES encodings shape: {x.shape}")
    smiles_list = featurnizer.decode(x)
    print(f"SMILES list: {smiles_list}")