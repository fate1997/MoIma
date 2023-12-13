import torch
from torch import Tensor, nn

from moima.dataset.smiles_seq.data import SeqBatch
from moima.model._util import init_weight
from moima.model.vae.decoders import GRUDecoder
from moima.model.vae.encoders import GRUEncoder


class ChemicalVAE(nn.Module):
    r"""Variational autoencoder for chemical data.
    
    Args:
        vocab_size (int): Vocabulary size.
        enc_hidden_dim (int): Encoder hidden dimension.
        latent_dim (int): Latent dimension.
        emb_dim (int): Embedding dimension.
        dec_hidden_dim (int): Decoder hidden dimension.
        dropout (float): Dropout rate.
    
    Structure:
        * Encoder: [batch_size, seq_len] -> [batch_size, seq_len, enc_hidden_dim]
        * Decoder: [batch_size, seq_len, latent_dim] -> [batch_size, seq_len, vocab_size]
    """
    def __init__(self, 
                 vocab_size: int=35,
                 enc_hidden_dim: int=292,
                 latent_dim: int=292,
                 emb_dim: int=128,
                 dec_hidden_dim: int=501,
                 dropout: float=0.1):
        super().__init__()
        self.encoder = GRUEncoder(vocab_size, 
                               emb_dim, 
                               enc_hidden_dim,
                               latent_dim,
                               dropout)
        self.decoder = GRUDecoder(self.encoder.embedding, 
                               dropout, 
                               latent_dim, 
                               dec_hidden_dim, 
                               vocab_size, 
                               emb_dim)
        self.apply(init_weight)
        
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        r"""Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_repr(self, batch: SeqBatch) -> Tensor:
        r"""Get latent representation of the input."""
        self.eval()
        mu, _ = self.encoder(batch)
        return mu

    def forward(self, batch: SeqBatch):
        r"""Forward pass of :class:`ChemicalVAE`.
        
        Args:
            batch (SeqBatch): Batch of data. The batch should contain :obj:`x` 
                and :obj:`seq_len`. The shape of :obj:`x` is :math:`[batch\_size, seq\_len]`. 
                The shape of :obj:`seq_len` is :math:`[batch\_size]`.
        
        Returns:
            mu (Tensor): Mean of the latent space. The shape is 
                :math:`[batch\_size, latent\_dim]`.
            logvar (Tensor): Log variance of the latent space. The shape is 
                :math:`[batch\_size, latent\_dim]`.
            y (Tensor): Output of the decoder. The shape is 
                :math:`[batch\_size, seq\_len, vocab\_size]`.
        """
        mu, logvar = self.encoder(batch)
        z = self.reparameterize(mu, logvar)
        y = self.decoder(z, batch)
        return y, mu, logvar