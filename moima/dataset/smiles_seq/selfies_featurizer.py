import pickle
import warnings
from copy import deepcopy
from typing import Set, List
import selfies as sf

import torch
from rdkit import Chem
from tqdm import tqdm

from moima.dataset._abc import FeaturizerABC
from moima.dataset.smiles_seq.data import SeqData

DEFAULT_VOCAB = {'[N]', '[O]'}


class SelfiesFeaturizer(FeaturizerABC):
    r"""The class for featurizing SELFIES strings into sequences.
    
    Args:
        vocab: A list of characters in the vocabulary.
        seq_len: The length of the sequence.
        DOUBLE_TOKEN_DICT: A dictionary of double tokens.
        SOS: The start of sequence token.
        EOS: The end of sequence token.
        PAD: The padding token.
    """
    
    # Special tokens
    SOS = '[$]'
    EOS = '[!]'
    PAD = '[nop]'
        
    def __init__(self, 
                 vocab: Set[str]=DEFAULT_VOCAB,
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
        r"""Encode a SELFIES string into a sequence."""
        try:
            selfies = sf.encoder(mol)
        except sf.EncoderError:
            print(f"SELFIES failed to encode SMILES: {mol}")
            return None
        selfies_symbols = list(sf.split_selfies(selfies))
        # Add special tokens (start, end, pad)
        if len(selfies_symbols) > self.seq_len - 2:
            selfies_symbols = selfies_symbols[:self.seq_len - 2]
            warnings.warn(f"SELFIES string {selfies} is longer than the maximum.")
        revised_selfies = f"{self.SOS}{selfies}{self.EOS}"
        
        seq = sf.selfies_to_encoding(revised_selfies,
                                     self.vocab_dict,
                                     pad_to_len=self.seq_len,
                                     enc_type='label')
        seq = torch.tensor(seq, dtype=torch.long)
        seq_len = torch.tensor(len(selfies_symbols) + 2, dtype=torch.long)
        smiles = self.decode(seq, is_raw=True)
        return SeqData(seq, seq_len, smiles)

    def reload_vocab(self, smiles_list: List[str]):
        r"""Reload the vocab by the given list of SMILES strings.
        
        Args:
            smiles_list (list): A list of SMILES strings.
        
        Returns:
            A list of uniqe characters in the SMILES strings.
        """
        selfies_list = []
        for smiles in tqdm(smiles_list, desc='Update vocabulary'):
            try:
                selfies = sf.encoder(smiles)
                selfies = f'{self.SOS}{selfies}{self.EOS}'
                selfies_list.append(selfies)
            except:
                pass
        vocab = sf.get_alphabet_from_selfies(selfies_list)
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
        vocab_idx = vocab_idx.cpu().numpy()
        selfies = sf.encoding_to_selfies(vocab_idx, self.idx2char, enc_type='label')
        # Tokens clear
        selfies = selfies.replace(self.SOS, '').replace(self.EOS, '').replace(self.PAD, '')
        try:
            smiles = sf.decoder(selfies)
        except sf.DecoderError:
            smiles = ''
        return smiles  
    
    def _set_vocab(self, vocab: Set[str]):
        r"""Set the vocab dictionary."""
        self.vocab = vocab
        self.vocab.add(self.PAD)
        self.vocab_dict =  {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {v: k for k, v in self.vocab_dict.items()}
       
    @property
    def arg4model(self) -> dict:
        return {'vocab_size': self.vocab_size}