import torch
import torch.nn.functional as F
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
    def __init__(
        self, 
        vocab_size: int=35,
        enc_hidden_dim: int=292,
        enc_num_layers: int=1,
        latent_dim: int=292,
        emb_dim: int=128,
        dec_hidden_dim: int=501,
        dec_num_layers: int=1,
        dropout: float=0.1,
        consider_label: bool=False,
        num_classes: int=0
    ):
        super().__init__()
        self.encoder = GRUEncoder(
            vocab_size, 
            emb_dim, 
            enc_hidden_dim,
            enc_num_layers,
            latent_dim,
            dropout,
            num_classes=num_classes
        )
        self.decoder = GRUDecoder(
            self.encoder.embedding, 
            dropout, 
            latent_dim+num_classes, 
            dec_hidden_dim, 
            dec_num_layers,
            vocab_size, 
            emb_dim
        )
        self.consider_label = consider_label
        if consider_label:
            self.label_embedding = nn.Embedding(3, emb_dim)
            self.label_embedding.apply(init_weight)
            self.nanlabel_embedding = nn.Embedding(2, emb_dim)
            self.nanlabel_embedding.weight.data = torch.stack(
                [torch.zeros(emb_dim), torch.ones(emb_dim)]
            ).to(self.nanlabel_embedding.weight.device)
            self.nanlabel_embedding.weight.requires_grad = False
        self.num_classes = num_classes
        if self.num_classes != 0:
            self.label_embedding = nn.Embedding(num_classes, emb_dim)
        
        self.apply(init_weight)
        
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        r"""Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def conditional_z(self, z, y) -> Tensor:
        r"""Get latent representation of the input."""
        self.eval()
        y = F.one_hot(y, self.num_classes).float()
        z = torch.cat([z, y], dim=-1)
        return z
    
    def get_label_emb(self, y) -> Tensor:
        r"""Get label embedding."""
        label_emb = self.label_embedding(y)
        if self.num_classes != None:
            return label_emb
        nanlabels = (y == 2).long()
        nanlabel_emb = self.nanlabel_embedding(nanlabels)
        emb = label_emb * nanlabel_emb
        return emb
    
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
        if self.consider_label:
            y = batch.y
            y = F.one_hot(y, self.num_classes).float()
        else:
            y = None
        mu, logvar = self.encoder(batch, y)
        z = self.reparameterize(mu, logvar)
        if self.consider_label:
            z = torch.cat([z, y], dim=-1)
        y = self.decoder(z, batch)
        return y, mu, logvar