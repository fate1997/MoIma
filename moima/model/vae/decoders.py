from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F


class GRUDecoder(nn.Module):
    def __init__(self, 
                 embedding: nn.Embedding,
                 emb_dropout=0.2,
                 latent_dim=292,
                 hidden_dim=501,
                 vocab_dim=35,
                 emb_dim=128):
        super().__init__()
        
        self.embedding = embedding
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.gru = nn.GRU(latent_dim+emb_dim, hidden_dim, batch_first=True)
        self.z2h = nn.Linear(latent_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_dim)
    
    def forward(self, z: torch.Tensor, seq: torch.Tensor, seq_len: torch.Tensor):
        input_emb = self.emb_dropout(self.embedding(seq))
        z_0 = z.unsqueeze(1).repeat(1, input_emb.size(1), 1)
        x_input = torch.cat([input_emb, z_0], dim=-1)
        packed_input = pack_padded_sequence(x_input, 
                                            seq_len.tolist(), 
                                            batch_first=True, 
                                            enforce_sorted=False)
        h_0 = self.z2h(z)
        h_0 = h_0.unsqueeze(0).repeat(1, 1, 1)
        output, _ = self.gru(packed_input, h_0)
        packed_output, _ = pad_packed_sequence(output, batch_first=True)
        y = self.output_head(packed_output)
        
        return y