import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUEncoder(nn.Module):
    def __init__(self, 
                 vocab_dim=35,
                 emb_dim=128,
                 enc_hidden_dim=292,
                 dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_dim, emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, enc_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, enc_hidden_dim)
    
    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor):
        input_emb = self.emb_dropout(self.embedding(seq))
        packed_input = pack_padded_sequence(input_emb, 
                                            seq_len.tolist(), 
                                            batch_first=True, 
                                            enforce_sorted=False)
        _, hidden = self.gru(packed_input)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = self.fc(hidden)
        return hidden
