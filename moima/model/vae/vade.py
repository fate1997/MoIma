import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import nn

from .decoders import GRUDecoder
from .encoders import GRUEncoder


class VaDE(nn.Module):
    """Variational Deep Embedding (VaDE) model. This implementation is referred to: 
        https://github.com/GuHongyang/VaDE-pytorch/blob/master/model.py, 
        https://github.com/zll17/Neural_Topic_Models/blob/6d8f0ce750393de35d3e0b03eae43ba39968bede/models/vade.py#L4
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
                               dropout)
        self.decoder = GRUDecoder(self.encoder.embedding, 
                               dropout, 
                               latent_dim, 
                               dec_hidden_dim, 
                               vocab_size, 
                               emb_dim)
        
        self.h2mu = nn.Linear(enc_hidden_dim, latent_dim)
        self.h2logvar = nn.Linear(enc_hidden_dim, latent_dim)
        
        self.pi_=nn.Parameter(torch.FloatTensor(n_clusters,).fill_(1)/n_clusters, requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(n_clusters, latent_dim).fill_(0), requires_grad=True)
        self.logvar_c=nn.Parameter(torch.FloatTensor(n_clusters, latent_dim).fill_(0), requires_grad=True)
        
        self.gmm = GaussianMixture(n_components=n_clusters, 
                                   covariance_type='diag')
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.apply(weight_init)
    
    @staticmethod
    def predict(model, batch):
        model.eval()
        seq, seq_len = batch.x, batch.seq_len
        hidden = model.encoder(seq, seq_len)
        z, _ = model.h2mu(hidden), model.h2logvar(hidden)

        pi = model.pi_
        log_sigma2_c = model.logvar_c
        mu_c = model.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+model.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)

        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_repr(self, batch):
        seq, seq_len = batch.x, batch.seq_len
        with torch.no_grad():
            hidden = self.encoder(seq, seq_len)
            mu = self.h2mu(hidden)
            return mu
    
    def forward(self, batch, is_pretrain: bool=False):
        seq, seq_len = batch.x, batch.seq_len
        hidden = self.encoder(seq, seq_len)
        mu, logvar = self.h2mu(hidden), self.h2logvar(hidden)
        if is_pretrain == False:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        x_hat = self.decoder(z, seq, seq_len)
        return x_hat, mu, logvar
    
    def ELBO_Loss(self, batch, L=1):
        det = 1e-10
        L_rec = 0
        
        # Get latent representation
        seq, seq_len = batch.x, batch.seq_len
        latent_output = self.encoder(seq, seq_len)
        z_mu, z_sigma2_log = self.h2mu(latent_output), self.h2logvar(latent_output)
        
        # Get reconstruction loss by Monte Carlo sampling
        for l in range(L):
            z = self.reparameterize(z_mu, z_sigma2_log)
            x_hat = self.decoder(z, seq, seq_len)
            recon_loss = F.cross_entropy(x_hat[:, :-1].contiguous().view(-1, x_hat.size(-1)),
                                    seq[:, 1:torch.max(seq_len).item()].contiguous().view(-1),
                                    ignore_index=0)
            L_rec += recon_loss
        L_rec /= L
        recon_loss = L_rec*batch.x.size(1)
        
        # Auxiliary variables
        Loss = L_rec*batch.x.size(1)
        pi =self.pi_
        log_sigma2_c=self.logvar_c
        mu_c=self.mu_c

        # Compute the posterior probability of z given c
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita_c=yita_c/(yita_c.sum(1).view(-1,1))
        self.yita_c=yita_c

        # Add KL divergence loss
        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))

        return x_hat, recon_loss, Loss - recon_loss
    
    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G=[]
        for c in range(self.n_clusters):
            G.append(self.gaussian_pdf_log(x, 
                                           mus[c:c+1,:], 
                                           log_sigma2s[c:c+1,:]).view(-1, 1))
        return torch.cat(G, dim=1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

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
        

if __name__ == '__main__':
    model = VaDE()
    seq = torch.randint(0, 35, (10, 100))
    seq_len = torch.randint(10, 100, (10,))
    out, mu, logvar, qc = model(seq, seq_len)
    print(out.shape)
    print(mu.shape)
    print(qc.shape)
    print(model.gmm_kl_div(mu, logvar))
    print(model.mus_mutual_distance())