import pickle
import warnings
from copy import deepcopy
from typing import List

import torch
from rdkit import Chem
from tqdm import tqdm

from moima.dataset._abc import FeaturizerABC
from moima.dataset.smiles_seq.data import SeqData

DEFAULT_VOCAB = [' ', '$', '!', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4',
 '5', '6', '7', '8', '=', '@', 'C', 'F', 'G', 'H', 'I', 'N', 'O', 'P', 'R', 'S', '[',
 '\\', ']', 'c', 'n', 'o', 's', '.']


class SeqFeaturizer(FeaturizerABC):
    r"""The class for featurizing SMILES strings into sequences.
    
    Args:
        vocab: A list of characters in the vocabulary.
        seq_len: The length of the sequence.
        DOUBLE_TOKEN_DICT: A dictionary of double tokens.
        SOS: The start of sequence token.
        EOS: The end of sequence token.
        PAD: The padding token.
    """
    
    # Double tokens
    DOUBLE_TOKEN_DICT = {
        'Br': 'R',
        'Cl': 'G',
        'Si': 'X',
    }
    
    # Special tokens
    SOS = '$'
    EOS = '!'
    PAD = ' '
        
    def __init__(self, 
                 vocab: List[str]=DEFAULT_VOCAB,
                 seq_len: int=120):
        assert len(set(vocab)) == len(vocab), "Vocabulary contains duplicate characters."
        self.seq_len = seq_len
        self._set_vocab(vocab)
    
    @property
    def vocab_size(self) -> int:
        r"""Return the size of the vocabulary."""
        return len(self.vocab)
    
    def __repr__(self) -> str:
        return f"SeqFeaturizer(seq_len: {self.seq_len}, vocab_size: {self.vocab_size})"
    
    def encode(self, mol: str) -> SeqData:
        r"""Encode a SMILES string into a sequence."""
        smiles_copy = deepcopy(mol)
        # Replace double tokens to single tokens
        smiles = self._double2single(mol)
                
        # Add special tokens (start, end, pad)
        if len(smiles) > self.seq_len - 2:
            smiles = smiles[:self.seq_len - 2]
            warnings.warn(f"SMILES string {smiles} is longer than the maximum.")
        revised_smiles = f"{self.SOS}{smiles}{self.EOS}"
        revised_smiles = revised_smiles.ljust(self.seq_len, self.PAD)
        
        # Encode SMILES
        try:
            seq = list(map(lambda x: self.vocab_dict[x], revised_smiles))
        except KeyError:
            return None
            raise KeyError(f"SMILES string {smiles_copy} contains unknown characters.")
        seq = torch.tensor(seq, dtype=torch.long)
        seq_len = torch.tensor(len(smiles) + 2, dtype=torch.long)
        return SeqData(seq, seq_len, smiles_copy)

    def reload_vocab(self, smiles_list: List[str]):
        r"""Reload the vocab by the given list of SMILES strings.
        
        Args:
            smiles_list (list): A list of SMILES strings.
        
        Returns:
            A list of uniqe characters in the SMILES strings.
        """
        s = set()
        for smiles in tqdm(smiles_list, desc='Update vocabulary'):
            smiles = Chem.CanonSmiles(smiles)
            smiles = self._double2single(smiles)
            for c in smiles:
                s.add(c)
        vocab = sorted(list(s))
        vocab = [self.PAD, self.SOS, self.EOS] + vocab
        self._set_vocab(vocab)
    
    def load_vocab(self, file_path: str) -> List[str]:
        r"""Load the vocab from a pickle file."""
        with open(file_path, 'rb') as f:
            vocab = pickle.load(f)
        self._set_vocab(vocab)
        return vocab
    
    def save_vocab(self, file_path: str) -> str:
        r"""Save the vocab to a pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.vocab, f)
        return file_path
    
    def decode(self, x: torch.Tensor, is_raw: bool=True) -> List[str]:
        r"""Decode SMILES encodings into a SMILES list.
        
        Args:
            x (torch.Tensor): SMILES encoding, shape of [pad_length, 
                vocab_length].
        
        Returns:
            A list of SMILES strings.
        """
        if is_raw:
            vocab_idx = x
        else:
            vocab_idx = torch.argmax(x, dim=1)
        smiles = "".join(map(lambda x: self.vocab[x], vocab_idx)).strip()
        # Tokens clear
        start = smiles.find(self.SOS)
        end = smiles.find(self.EOS)
        smiles = smiles[start + 1:end]
        smiles = self._single2double(smiles)
        return smiles  
    
    def _set_vocab(self, vocab: List[str]):
        r"""Set the vocab dictionary."""
        self.vocab = vocab
        self.vocab_dict =  {c: i for i, c in enumerate(self.vocab)}
    
    def _double2single(self, smiles: str) -> str:
        r"""Replace double tokens to single tokens."""
        for k, v in self.DOUBLE_TOKEN_DICT.items():
            smiles = smiles.replace(k, v)
        return smiles
    
    def _single2double(self, smiles: str) -> str:
        r"""Replace single tokens to double tokens."""
        for k, v in self.DOUBLE_TOKEN_DICT.items():
            smiles = smiles.replace(v, k)
        return smiles
    
    @property
    def arg4model(self) -> dict:
        return {'vocab_size': self.vocab_size}