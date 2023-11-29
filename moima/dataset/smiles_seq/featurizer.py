import warnings
from typing import Any, List

import torch
from rdkit import Chem

from moima.dataset._abc import FeaturizerABC
from moima.dataset.smiles_seq.data import SeqData

DEFAULT_CHARSET = [' ', '$', '!', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4',
 '5', '6', '7', '8', '=', '@', 'C', 'F', 'G', 'H', 'I', 'N', 'O', 'P', 'R', 'S', '[',
 '\\', ']', 'c', 'n', 'o', 's', '.']


class SeqFeaturizer(FeaturizerABC):
    DOUBLE_TOKEN_DICT = {
        'Br': 'R',
        'Cl': 'G',
        'Si': 'X'
    }
    # Special tokens
    SOS = '$'
    EOS = '!'
    PAD = ' '
        
    def __init__(self, 
                 charset: List[str]=DEFAULT_CHARSET,
                 seq_len: int=120):
        self.seq_len = seq_len
        self.charset = charset
        self.set_charset_dict()
    
    def __repr__(self) -> str:
        return f"SeqFeaturizer(seq_len: {self.seq_len})"
    
    @property
    def __dict__(self) -> dict:
        return {
            'charset': self.charset,
            'seq_len': self.seq_len,
            'vocab_size': self.vocab_size,
        }
    
    def set_charset_dict(self):
        self.charset_dict =  {c: i for i, c in enumerate(self.charset)}
    
    @property
    def input_args(self):
        dic = {
            'charset': self.charset,
            'seq_len': self.seq_len
        }
        return dic
    
    def __call__(self, smiles: str) -> SeqData:
        """Featurize a SMILES string into a sequence."""
        # Replace double tokens to single tokens
        for k, v in self.DOUBLE_TOKEN_DICT.items():
            smiles = smiles.replace(k, v)
                
        # Add start and end tokens
        if len(smiles) > self.seq_len - 2:
            smiles = smiles[:self.seq_len - 2]
            warnings.warn(f"SMILES string {smiles} is longer than the maximum.")

        smiles = f"{self.SOS}{smiles}{self.EOS}"
        paded_smiles = smiles.ljust(self.seq_len, self.PAD)
        seq = list(map(lambda x: self.charset_dict[x], paded_smiles))
        seq = torch.tensor(seq, dtype=torch.long)
        return SeqData(seq, smiles)
    
    def reload_charset(self, smiles_list: List[str]) -> List[str]:
        r"""Reload the charset by the given list of SMILES strings.
        
        Args:
            smiles_list (list): A list of SMILES strings.
        
        Returns:
            A list of uniqe characters in the SMILES strings.
        """
        s = set()
        for smiles in smiles_list:
            smiles = Chem.CanonSmiles(smiles)
            # Replace double tokens to single tokens
            for k, v in self.DOUBLE_TOKEN_DICT.items():
                smiles = smiles.replace(k, v)
            for c in smiles:
                s.add(c)
        charset = sorted(list(s))
        charset = [self.PAD, self.SOS, self.EOS] + charset
        self.charset = charset
        self.set_charset_dict()
    
    def decode(self, x: torch.Tensor, is_raw: bool=True) -> List[str]:
        r"""Decode SMILES encodings into a SMILES list.
        
        Args:
            x (torch.Tensor): SMILES encoding, shape of [pad_length, 
                charset_length].
        
        Returns:
            A list of SMILES strings.
        """
        if is_raw:
            charset_idx = x[:-1]
        else:
            charset_idx = torch.argmax(x, dim=1)
        smiles = "".join(map(lambda x: self.charset[x], charset_idx)).strip()
        # Tokens clear
        start = smiles.find(self.SOS)
        end = smiles.find(self.EOS)
        smiles = smiles[start + 1:end]
        for k, v in self.DOUBLE_TOKEN_DICT.items():
            smiles = smiles.replace(v, k)
        return smiles
    
    @property
    def vocab_size(self):
        return len(self.charset)


if __name__ == '__main__':
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CCO',
        'CC(=O)O',
    ]
    featurizer = SeqFeaturizer(seq_len=256)
    data = featurizer(smiles_list[0])
    print(f"SMILES encodings shape: {data.x.shape}")
    smiles = featurizer.decode(data.x, is_raw=True)
    print(f"SMILES input:  {smiles_list[0]}")
    print(f"SMILES output: {smiles}")