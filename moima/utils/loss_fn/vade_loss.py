import torch
import torch.nn.functional as F
import numpy as np


class VaDELossCalc:
    def __init__(self, 
                 start_kl_weight: float = 0.0001,
                 end_kl_weight: float = 0.0025,
                 anneal_num_epochs: int = 100,
                 n_cycles: int = 5,
                 ratio: float = 0.7,
                 start_center_weight: float = 0.00001,
                 end_center_weight: float = 1e7):
        self.kl_scheduler = self.loss_annealing(start_kl_weight, 
                                                end_kl_weight, 
                                                anneal_num_epochs, 
                                                n_cycles, 
                                                ratio)
        self.center_scheduler = self.loss_annealing(start_center_weight,
                                                    end_center_weight,
                                                    anneal_num_epochs,
                                                    n_cycles,
                                                    ratio)
    
    def loss_annealing(self, 
                       start_weight: float = 0.00001,
                        end_weight: float = 0.0025,
                        num_epochs: int = 100,
                        n_cycles: int = 5,
                        ratio: float = 0.7):#
        # Get the KL weight schedule
        scheduler = end_weight * np.ones(num_epochs)
        period = num_epochs / n_cycles
        step = (end_weight - start_weight)/(period * ratio)

        for c in range(n_cycles):
            v , i = start_weight, 0
            while v <= end_weight and (int(i+c * period) < num_epochs):
                scheduler[int(i+c * period)] = v
                v += step
                i += 1
        return scheduler
            
    def __call__(self, batch, model, current_epoch):
        x_hat, recon_loss, kl_loss = model.ELBO_Loss(batch)
        kl_scheduler = self.kl_scheduler[current_epoch]
        kl_loss = kl_scheduler * kl_loss
        return x_hat, recon_loss, kl_loss

    
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