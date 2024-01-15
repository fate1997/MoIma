from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import Tensor, nn
from torch.distributions.multivariate_normal import MultivariateNormal

from moima.dataset.smiles_seq.data import SeqBatch
from moima.model._util import init_weight
from moima.model.vae.decoders import GRUDecoder
from moima.model.vae.encoders import GRUEncoder


class VaDE(nn.Module):
    """Variational Deep Embedding (VaDE) model. This implementation is referred to: 
        https://github.com/GuHongyang/VaDE-pytorch/blob/master/model.py, 
        https://github.com/zll17/Neural_Topic_Models/blob/6d8f0ce750393de35d3e0b03eae43ba39968bede/models/vade.py#L4
    
    Args:
        vocab_size (int): Vocabulary size.
        enc_hidden_dim (int): Encoder hidden dimension.
        latent_dim (int): Latent dimension.
        emb_dim (int): Embedding dimension.
        dec_hidden_dim (int): Decoder hidden dimension.
        n_clusters (int): Number of clusters.
        dropout (float): Dropout rate.
    
    Structure:
        * Encoder: [batch_size, seq_len] -> [batch_size, seq_len, enc_hidden_dim]
        * Decoder: [batch_size, seq_len, latent_dim] -> [batch_size, seq_len, vocab_size]
    
    Additional Attributes:
        * pi_ (nn.Parameter): Mixture weight.
        * mu_c (nn.Parameter): Cluster mean.
        * logvar_c (nn.Parameter): Cluster log variance.
        * gmm (sklearn.mixture.GaussianMixture): Gaussian mixture model.
        * n_clusters (int): Number of clusters.
        * latent_dim (int): Latent dimension.
    """
    def __init__(self, 
                 vocab_size: int=35,
                 enc_hidden_dim: int=292,
                 latent_dim: int=292,
                 emb_dim: int=128,
                 dec_hidden_dim: int=501,
                 n_clusters: int=10,
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
        
        # Additional parameters to define the Gaussian mixture model
        self.pi_unnorm = nn.Parameter(torch.FloatTensor(n_clusters, ).fill_(1)/n_clusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(n_clusters, latent_dim).fill_(0), requires_grad=True)
        self.logvar_c = nn.Parameter(torch.FloatTensor(n_clusters, latent_dim).fill_(0), requires_grad=True)
        self.gmm = GaussianMixture(n_components=n_clusters,
                                   covariance_type='diag', 
                                   random_state=3)
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        
        self.apply(init_weight)
    
    @staticmethod
    def inverse_softmax(x: Tensor, c: float=0) -> Tensor:
        return torch.log(x) + c
    
    @property
    def pi_(self):
        return F.softmax(self.pi_unnorm, dim=0)
    
    def predict(self, batch: SeqBatch) -> np.ndarray:
        r"""Predict the cluster label of the input.
        
        Args:
            model (VaDE): VaDE model.
            batch (SeqBatch): Batch of data. The batch should contain :obj:`x` and :obj:`seq_len`.
                The shape of :obj:`x` is :math:`[batch\_size, seq\_len]`. The shape of :obj:`seq_len`
                is :math:`[batch\_size]`.
        
        Returns:
            y_pred (np.ndarray): Cluster label of the input. The shape is :math:`[batch\_size]`.
        """
        self.eval()
        z, _ = self.encoder(batch)
        pi = self.pi_
        logvar_c = self.logvar_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+
                           self.gaussian_pdfs_log(z, mu_c, logvar_c))
        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        r"""Reparameterization trick. """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_repr(self, batch: SeqBatch) -> Tensor:
        r"""Get the latent representation of the input."""
        self.eval()
        mu, _ = self.encoder(batch)
        return mu
    
    def forward(self, batch: SeqBatch, is_pretrain: bool=False) \
                            -> Tuple[Tensor, Tensor, Tensor]:
        r"""Forward pass of :class:`VaDE`. 
        
        Args:
            batch (SeqBatch): Batch of data. The batch should contain :obj:`x` and :obj:`seq_len`.
                The shape of :obj:`x` is :math:`[batch\_size, seq\_len]`. The shape of :obj:`seq_len`
                is :math:`[batch\_size]`.
            is_pretrain (bool): Whether to use the pretrain mode. (default: :obj:`False`)
        
        Returns:
            x_hat (Tensor): Reconstruction of the input. The shape is :math:`[batch\_size, seq\_len, vocab\_size]`.
            mu (Tensor): Mean of the latent space. The shape is :math:`[batch\_size, latent\_dim]`.
            logvar (Tensor): Log variance of the latent space. The shape is :math:`[batch\_size, latent\_dim]`.
            eta_c (Tensor): The probability of each cluster given z. The shape is :math:`[batch\_size, n\_clusters]`.
                $$\eta_c=\frac{\pi_c\mathcal{N}(z|\mu_c, \sigma_c)}{\sum_{c=1}^C\pi_c\mathcal{N}(z|\mu_c, \sigma_c)}$$
        """
        mu, logvar = self.encoder(batch)
        if is_pretrain == False:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        x_hat = self.decoder(z, batch)
        # Calculate the probability of each cluster given z
        det = 1e-10
        pi = F.softmax(self.pi_, dim=0)
        log_sigma2_c = self.logvar_c
        mu_c = self.mu_c
        z = self.reparameterize(mu, logvar)
        scale = torch.diag_embed(torch.exp(log_sigma2_c))
        normal_pdf = MultivariateNormal(mu_c, scale).log_prob(z.unsqueeze(1))
        eta_c = torch.log(pi.unsqueeze(0)) + normal_pdf
        log_eta_c = eta_c - torch.logsumexp(eta_c, dim=1, keepdim=True)

        if torch.isnan(log_eta_c).any():
            print(f'eta_c has nan values: {torch.isnan(eta_c).any()}')
            print(f'min_pi: {pi.min().item()}')
            print(f'log_eta_c: {log_eta_c}')
            raise ValueError('eta_c contains NaN')

        return x_hat, mu, logvar, log_eta_c
    
    def ELBO_Loss(self, batch: SeqBatch) -> Dict[str, Tensor]:
        r"""Calculate the ELBO loss of :class:`VaDE`.
        
        Args:
            batch (SeqBatch): Batch of data. The batch should contain :obj:`x` and :obj:`seq_len`.
                The shape of :obj:`x` is :math:`[batch\_size, seq\_len]`. The shape of :obj:`seq_len`
                is :math:`[batch\_size]`.
            num_mc_rounds (int): Number of Monte Carlo sampling when calculating the reconstruction loss.
                (default: :obj:`1`)

        Returns:
            x_hat (Tensor): Reconstruction of the input. The shape is :math:`[batch\_size, seq\_len, vocab\_size]`.
            recon_loss (Tensor): Reconstruction loss. Calculated by:
                $$\mathcal{L}_{rec}=-\sum_{t=1}^{T}\log p(x_t|z)$$
            kl_loss (Tensor): KL divergence loss. Calculated by:
                $$\mathcal{L}_{KL}=\sum_{c=1}^{C}\sum_{i=1}^{N}q(c_i|x_i)\log \frac{q(c_i|x_i)}{p(c_i)}-\sum_{c=1}^{C}q(c|x_i)\log q(c|x_i)$$
        """
        det = 1e-10
        L_rec = 0
        
        # Get latent representation
        mu, logvar = self.encoder(batch)
        
        # Get reconstruction loss by Monte Carlo sampling
        z = self.reparameterize(mu, logvar)
        seq, seq_len = batch.x, batch.seq_len
        x_hat = self.decoder(z, batch)
        L_rec = F.cross_entropy(x_hat[:, :-1].contiguous().view(-1, x_hat.size(-1)),
                                seq[:, 1:torch.max(seq_len).item()].contiguous().view(-1),
                                ignore_index=0)
        recon_loss = L_rec * batch.x.size(1)
        
        # Auxiliary variables
        Loss = L_rec*batch.x.size(1)
        pi = self.pi_
        log_sigma2_c = self.logvar_c
        mu_c = self.mu_c

        # Compute the posterior probability of z given c
        z = self.reparameterize(mu, logvar)
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita_c = yita_c/(yita_c.sum(1).view(-1,1))

        # Add KL divergence loss
        Loss += 0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(logvar.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))
        # print(f'first term: {0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+torch.exp(logvar.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+(mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1)).item()}, second term: {(torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+logvar,1))).item()}')
        Loss -= torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+logvar,1))

        return x_hat, recon_loss, Loss - recon_loss