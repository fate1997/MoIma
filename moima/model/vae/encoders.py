from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from moima.dataset.smiles_seq.data import SeqBatch
from moima.model._util import init_weight


class GRUEncoder(nn.Module):
    r"""Gate Recurrent Unit (GRU) encoder.
    
    Args:
        vocab_dim (int): Vocabulary dimension.
        emb_dim (int): Embedding dimension.
        enc_hidden_dim (int): Encoder hidden dimension.
        dropout (float): Dropout rate.
    
    Structure:
        * Embedding: [batch_size, seq_len] -> [batch_size, seq_len, emb_dim]
        * GRU: [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, enc_hidden_dim]
        * Linear: [batch_size, seq_len, enc_hidden_dim] -> [batch_size, seq_len, enc_hidden_dim]
    """
    def __init__(self, 
                 vocab_dim: int=35,
                 emb_dim: int=128,
                 enc_hidden_dim: int=292,
                 num_layers: int=1,
                 latent_dim: int=292,
                 dropout :float=0.2,
                 num_classes: int=0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_dim, emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim + num_classes, 
                          enc_hidden_dim, 
                          num_layers, 
                          batch_first=True, 
                          bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, enc_hidden_dim)
        self.h2mu=nn.Linear(enc_hidden_dim, latent_dim)
        self.h2logvar=nn.Linear(enc_hidden_dim, latent_dim)
    
    def forward(self, batch: SeqBatch, y: Tensor=None) -> Tuple[Tensor, Tensor]:
        r"""Forward pass of :class:`GRUEncoder`.
        
        Args:
            batch (SeqBatch): Batch of data. The batch should contain :obj:`x` and :obj:`seq_len`.
                The shape of :obj:`x` is :math:`[batch\_size, seq\_len]`. The shape of :obj:`seq_len`
                is :math:`[batch\_size]`.
        
        Returns:
            mu (Tensor): Mean of latent space.
            logvar (Tensor): Log variance of latent space.
        """
        seq, seq_len = batch.x, batch.seq_len
        input_emb = self.emb_dropout(self.embedding(seq))
        if y is not None:
            y = y.unsqueeze(1).expand(-1, seq.size(1), -1)
            input_emb = torch.cat([input_emb, y], dim=-1)
        packed_input = pack_padded_sequence(input_emb, 
                                            seq_len.tolist(), 
                                            batch_first=True, 
                                            enforce_sorted=False)
        _, hidden = self.gru(packed_input)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = self.fc(hidden)
        mu, logvar = self.h2mu(hidden), self.h2logvar(hidden)
        return mu, logvar