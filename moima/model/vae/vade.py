import numpy as np
import torch
import torch.nn.functional as F
from .decoders import GRUDecoder
from .encoders import GRUEncoder
from sklearn.mixture import GaussianMixture
from torch import nn


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
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_latent(self, seq: torch.Tensor, seq_len: torch.Tensor):
        with torch.no_grad():
            hidden = self.encoder(seq, seq_len)
            mu = self.h2mu(hidden)
            return mu
    
    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor, is_pretrain: bool=False):
        
        hidden = self.encoder(seq, seq_len)
        mu, logvar = self.h2mu(hidden), self.h2logvar(hidden)
        if is_pretrain == False:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
             
        x_hat = self.decoder(z, seq, seq_len)
        return x_hat, mu, logvar
    
    def ELBO_Loss(self,batch,L=1):
        det=1e-10

        L_rec=0
        seq, seq_len = batch.x, batch.seq_len
        latent_output = self.encoder(seq, seq_len)
        z_mu, z_sigma2_log = self.h2mu(latent_output), self.h2logvar(latent_output)
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_hat = self.decoder(z, seq, seq_len)

            recon_loss = F.cross_entropy(x_hat[:, :-1].contiguous().view(-1, x_hat.size(-1)),
                                    seq[:, 1:torch.max(seq_len).item()].contiguous().view(-1),
                                    ignore_index=0)
            L_rec+=recon_loss

        L_rec/=L

        Loss=L_rec*batch.x.size(1)

        pi=self.pi_
        log_sigma2_c=self.logvar_c
        mu_c=self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))

        return x_hat, Loss    
    
    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.n_clusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

    def gmm_kl_div(self, mu: torch.Tensor, logvar: torch.Tensor):
        zs = self.reparameterize(mu, logvar)
        mu_c = self.mu_c
        logvar_c = self.logvar_c
        delta = 1e-10
        print(f'zs range: {zs.min().item()} - {zs.max().item()}, mu_c range: {mu_c.min().item()} - {mu_c.max().item()}')
        gamma_c = torch.exp(torch.log(self.pi.unsqueeze(0))+self.log_pdfs_gauss(zs,mu_c,logvar_c))+delta
        print(f'raw gamma_c range: {gamma_c.min().item()} - {gamma_c.max().item()}')
        gamma_c = gamma_c / (gamma_c.sum(dim=1).view(-1,1))

        kl_div = 0.5 * torch.mean(torch.sum(gamma_c*torch.sum(logvar_c.unsqueeze(0)+
                        torch.exp(logvar.unsqueeze(1)-logvar_c.unsqueeze(0))+
                        (mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/(torch.exp(logvar_c.unsqueeze(0))),dim=2),dim=1))
        print(f'gamma_c range: {gamma_c.min().item()} - {gamma_c.max().item()}, pi range: {self.pi.min().item()} - {self.pi.max().item()}')
        kl_div -= torch.mean(torch.sum(gamma_c*torch.log(self.pi.unsqueeze(0)/gamma_c),dim=1)) + \
                  0.5*torch.mean(torch.sum(1+logvar,dim=1))
        return kl_div

    def log_pdfs_gauss(self,z,mus,logvars):
        # Compute log value of the posterion probability of z given mus and logvars under GMM hypothesis.
        # i.e. log(p(z|c)) in the Equation (16) (the second term) of the original paper.
        # params: z=[batch_size * latent_dim], mus=[n_clusters * latent_dim], logvars=[n_clusters * latent_dim]
        # return: [batch_size * n_clusters], each row is [log(p(z|c1)),log(p(z|c2)),...,log(p(z|cK))]
        log_pdfs = []
        for c in range(self.n_clusters):
            log_pdfs.append(self.log_pdf_gauss(z,mus[c:c+1,:],logvars[c:c+1,:]))
        return torch.cat(log_pdfs,dim=1)

    def log_pdf_gauss(self,z,mu,logvar):
        # Compute the log value of the probability of z given mu and logvar under gaussian distribution
        # i.e. log(p(z|c)) in the Equation (16) (the numerator of the last term) of the original paper
        # params: z=[batch_size * latent_dim], mu=[1 * latent_dim], logvar=[1 * latent_dim]
        # return: res=[batch_size,1], each row is the log val of the probability of a data point w.r.t the component N(mu,var)
        '''
            log p(z|c_k) &= -(J/2)log(2*pi) - (1/2)*\Sigma_{j=1}^{J} log sigma_{j}^2 - \Sigma_{j=1}^{J}\frac{(z_{j}-mu_{j})^2}{2*\sigma_{j}^{2}}
                         &=-(1/2) * \{[log2\pi,log2\pi,...,log2\pi]_{J}
                                    + [log\sigma_{1}^{2},log\sigma_{2}^{2},...,log\sigma_{J}^{2}]_{J}
                                    + [(z_{1}-mu_{1})^2/(sigma_{1}^{2}),(z_{2}-mu_{2})^2/(sigma_{2}^{2}),...,(z_{J}-mu_{J})^2/(sigma_{J}^{2})]_{J}                   
                                    \},
            where J = latent_dim
        '''
        return (-0.5*(torch.sum(np.log(2*np.pi)+logvar+(z-mu).pow(2)/torch.exp(logvar),dim=1))).view(-1,1)
    
    def mus_mutual_distance(self, dist_type='cosine'):
        if dist_type=='cosine':
            norm_mu = self.mu_c / torch.norm(self.mu_c,dim=1,keepdim=True)
            cos_mu = torch.matmul(norm_mu,norm_mu.transpose(1,0))
            cos_sum_mu = torch.sum(cos_mu) # the smaller the better
        
            theta = F.softmax(self.z2theta(self.mu_c),dim=1)
            cos_theta = torch.matmul(theta,theta.transpose(1,0))
            cos_sum_theta = torch.sum(cos_theta)
        
            dist = cos_sum_mu + cos_sum_theta
        else:
            mu = self.mu_c
            dist = torch.reshape(torch.sum(mu**2,dim=1),(mu.shape[0],1))+ torch.sum(mu**2,dim=1)-2*torch.matmul(mu,mu.t())
            dist = 1.0/(dist.sum() * 0.5) + 1e-12
        return dist
        
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