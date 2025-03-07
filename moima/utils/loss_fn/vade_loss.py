import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Dict
from moima.dataset.smiles_seq.data import SeqBatch
from moima.model.vae.vade import VaDE


class VaDELossCalc:
    r"""Calculate the loss of the VaDE model.
    
    Args:
        kl_weight (float): The weight of the KL divergence loss. Default to 1.0.
    """
    def __init__(self, kl_weight: float=1.0):
        self.kl_weight = kl_weight
        self.kl_scheduler = self.loss_annealing()
    
    def loss_annealing(self, 
                       start_weight: float = 0.0001,
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
            
    def __call__(self, 
                 batch: SeqBatch, 
                 mu: Tensor,
                 logvar: Tensor,
                 x_hat: Tensor,
                 log_eta_c: Tensor,
                 pi: Tensor, 
                 mu_c: Tensor,
                 logvar_c: Tensor,
                 current_epoch: int) -> Dict[str, Tensor]:
        r"""Calculate the loss of the VaDE model.
        
        Args:
            batch (SeqBatch): Batch of input data.
            mu (Tensor): The mean of the latent representation.
            logvar (Tensor): The log variance of the latent representation.
            x_hat (Tensor): Reconstruction of the input.
            log_eta_c (Tensor): The probability of c given z.
            pi (Tensor): The prior probability of c.
            mu_c (Tensor): The mean of the latent representation given c.
            logvar_c (Tensor): The log variance of the latent representation given c.
            current_epoch (int): The current epoch.
        
        Returns:
            A dictionary of loss. The keys are :obj:`recon_loss`, :obj:`kl_loss`, 
                and :obj:`loss`.
        """
        recon_loss = self.recon_loss(batch, x_hat)
        kl_loss = self.kl_loss(mu, logvar, log_eta_c, pi, mu_c, logvar_c)
        kl_loss = self.kl_scheduler[current_epoch] * kl_loss# self.kl_weight * kl_loss
        loss_dict = {'recon_loss': recon_loss, 
                     'kl_loss': kl_loss, 
                     'loss': recon_loss+kl_loss}
        return loss_dict
    
    @staticmethod
    def recon_loss(batch: SeqBatch, x_hat: Tensor) -> Tensor:
        r"""Calculate the reconstruction loss by the following formula:
            $$\sum_{t=1}^T \sum_{k=1}^K \pi_k \mathcal{N}(x_t|\mu_k, \sigma_k)$$
        
        Args:
            batch (SeqBatch): Batch of input data.
            x_hat (Tensor): Reconstruction of the input.
        
        Returns:
            The reconstruction loss.
        """
        seq, seq_len = batch.x, batch.seq_len
        loss = F.cross_entropy(x_hat[:, :-1].contiguous().view(-1, x_hat.size(-1)),
                        seq[:, 1:torch.max(seq_len).item()].contiguous().view(-1),
                        ignore_index=0)
        loss = loss * seq.size(1)
        return loss
    
    @staticmethod
    def kl_loss(mu: Tensor, 
                logvar: Tensor, 
                log_eta_c: Tensor,
                pi: Tensor, 
                mu_c: Tensor,
                logvar_c: Tensor) -> Tensor:
        r"""Calculate the KL divergence loss by the following formula:
            $$\sum_{c=1}^C \sum_{i=1}^N q(c_i|x_i) \log \frac{q(c_i|x_i)}{p(c_i)} - \sum_{c=1}^C q(c|x_i) \log q(c|x_i)$$
        
        Args:
            mu (Tensor): The mean of the latent representation.
            logvar (Tensor): The log variance of the latent representation.
            log_eta_c (Tensor): The probability of c given z.
            pi (Tensor): The prior probability of c.
            mu_c (Tensor): The mean of the latent representation given c.
            logvar_c (Tensor): The log variance of the latent representation given c.
        
        Returns:
            The KL divergence loss.
        """
        # Add KL divergence loss
        eta_c = torch.exp(log_eta_c)
        if torch.isnan(eta_c).any():
            print(eta_c)
            print(log_eta_c)
            raise ValueError('eta_c contains NaN')
        loss = 0.5*torch.mean(torch.sum(eta_c*torch.sum(logvar_c.unsqueeze(0)+
                                        torch.exp(logvar.unsqueeze(1)-logvar_c.unsqueeze(0))+
                                        (mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(logvar_c.unsqueeze(0)),2),1))
        if torch.isnan(loss).any():
            raise ValueError('loss contains NaN')
        # print(f'first term: {loss.item()}, second term: {(torch.mean(torch.sum(eta_c*torch.log(pi.unsqueeze(0)/(eta_c)),1))+0.5*torch.mean(torch.sum(1+logvar,1))).item()}')
        loss -= torch.mean(torch.sum(eta_c*(torch.log(pi.unsqueeze(0))-log_eta_c),1))+0.5*torch.mean(torch.sum(1+logvar,1))
        if torch.isnan(loss).any():
            print(f'** eta_c: {eta_c}, pi: {pi}, pi/eta_c: {pi.unsqueeze(0)/(eta_c)}')
            raise ValueError('loss (2nd) contains NaN')
        return loss