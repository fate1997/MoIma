import torch
from torch import nn
from torch.nn import functional as F

from moima.model.vae.decoders import GRUDecoder
from moima.model.vae.encoders import GRUEncoder


class ChemicalVAE(nn.Module):
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
                               dropout)
        self.decoder = GRUDecoder(self.encoder.embedding, 
                               dropout, 
                               latent_dim, 
                               dec_hidden_dim, 
                               vocab_size, 
                               emb_dim)
        self.h2mu = nn.Linear(enc_hidden_dim, latent_dim)
        self.h2logvar = nn.Linear(enc_hidden_dim, latent_dim)
        self.apply(weight_init)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_repr(self, batch):
        self.eval()
        seq, seq_len = batch.x, batch.seq_len
        hidden = self.encoder(seq, seq_len)
        mu = self.h2mu(hidden)
        return mu

    def forward(self, batch):
        seq, seq_len = batch.x, batch.seq_len
        hidden = self.encoder(seq, seq_len)
        mu, logvar = self.h2mu(hidden), self.h2logvar(hidden)
        z = self.reparameterize(mu, logvar)
        y = self.decoder(z, seq, seq_len)
        
        return mu, logvar, y


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        
        nn.init.orthogonal_(m.all_weights[0][0])
        nn.init.orthogonal_(m.all_weights[0][1])
        nn.init.zeros_(m.all_weights[0][2])
        nn.init.zeros_(m.all_weights[0][3])
        
    elif isinstance(m, nn.Embedding):
        nn.init.constant_(m.weight, 1)