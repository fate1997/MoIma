import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from moima.dataset.smiles_seq.data import SeqBatch


class GRUDecoder(nn.Module):
    r"""Gate Recurrent Unit (GRU) decoder.
    
    Args:
        embedding (nn.Embedding): Embedding layer from encoder.
        emb_dropout (float): Dropout rate.
        latent_dim (int): Latent dimension.
        hidden_dim (int): Hidden dimension.
        vocab_dim (int): Vocabulary dimension.
        emb_dim (int): Embedding dimension.
    
    Structure:
        * Embedding: [batch_size, seq_len] -> [batch_size, seq_len, emb_dim]
        * GRU: [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, hidden_dim]
        * Linear: [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, vocab_dim]
    """
    def __init__(
        self, 
        embedding: nn.Embedding,
        emb_dropout=0.2,
        latent_dim=292,
        hidden_dim=501,
        num_layers=1,
        vocab_dim=35,
        emb_dim=128
    ):
        super().__init__()
        
        self.embedding = embedding
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.gru = nn.GRU(latent_dim+emb_dim, hidden_dim, num_layers, batch_first=True)
        self.z2h = nn.Linear(latent_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_dim)
        self.num_layers = num_layers
    
    def forward(self, z: Tensor, batch: SeqBatch) -> Tensor:
        r"""Forward pass of :class:`GRUDecoder`.
        
        Args:
            z (Tensor): Latent space. The shape is :math:`[batch\_size, latent\_dim]`.
            batch (SeqBatch): Batch of data. The batch should contain :obj:`x` and :obj:`seq_len`.
                The shape of :obj:`x` is :math:`[batch\_size, seq\_len]`. The shape of :obj:`seq_len`
                is :math:`[batch\_size]`.
        
        Returns:
            y (Tensor): Output. The shape is :math:`[batch\_size, seq\_len, vocab\_dim]`.
        """
        seq, seq_len = batch.x, batch.seq_len
        input_emb = self.emb_dropout(self.embedding(seq))
        z_0 = z.unsqueeze(1).repeat(1, input_emb.size(1), 1)
        x_input = torch.cat([input_emb, z_0], dim=-1)
        packed_input = pack_padded_sequence(x_input, 
                                            seq_len.tolist(), 
                                            batch_first=True, 
                                            enforce_sorted=False)
        h_0 = self.z2h(z)
        h_0 = h_0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        output, _ = self.gru(packed_input, h_0)
        packed_output, _ = pad_packed_sequence(output, batch_first=True)
        y = self.output_head(packed_output)
        return y