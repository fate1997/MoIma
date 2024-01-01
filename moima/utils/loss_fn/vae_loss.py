from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from moima.dataset.smiles_seq.data import SeqBatch
from moima.utils._util import KL_divergence

from ._abc import LossCalcABC


class VAELossCalc(LossCalcABC):
    r"""A class to calculate the loss of VAE.
    
    Args:
        start_kl_weight (float): The initial KL weight. Default to 0.0.
        end_kl_weight (float): The final KL weight. Default to 0.0025.
        num_epochs (int): The number of epochs. Default to 100.
        n_cycles (int): The number of cycles. Default to 5.
        ratio (float): The ratio of the increasing period. Default to 0.7.
    
    Examples:
        >>> loss_calc = VAELossCalc()
        >>> loss = loss_calc(batch, mu, logvar, x_hat, current_epoch)
    """
    def __init__(self,
                 start_kl_weight: float = 0.0,
                 end_kl_weight: float = 0.0025,
                 num_epochs: int = 100,
                 n_cycles: int = 5,
                 ratio: float = 0.7):
        
        # Get the KL weight schedule
        kl_scheduler = end_kl_weight * np.ones(num_epochs)
        period = num_epochs / n_cycles
        step = (end_kl_weight - start_kl_weight)/(period * ratio)
        for c in range(n_cycles):
            v , i = start_kl_weight, 0
            while v <= end_kl_weight and (int(i+c * period) < num_epochs):
                kl_scheduler[int(i+c * period)] = v
                v += step
                i += 1
        self.kl_scheduler = kl_scheduler
    
    def __call__(self, 
                 batch: SeqBatch,
                 mu: Tensor,
                 logvar: Tensor,
                 x_hat: Tensor,
                 current_epoch: int) -> Dict[str, Tensor]:
        r"""Calculate the loss of ChemicalVAE.
        
        Args:
            batch (SeqBatch): A batch of sequences.
            mu (Tensor): The mean of the latent distribution.
            logvar (Tensor): The log variance of the latent distribution.
            x_hat (Tensor): The predicted sequences.
            current_epoch (int): The current epoch.
        
        Returns:
            A dictionary of losses. The keys are `recon_loss` and `kl_loss`.
        """
        loss_dict = {}
        kl_weight = self.kl_scheduler[current_epoch]
        kl_loss = kl_weight * KL_divergence(mu, logvar) / mu.size(0)
        seq, seq_len = batch.x, batch.seq_len
        recon_loss = F.cross_entropy(x_hat[:, :-1].contiguous().view(-1, x_hat.size(-1)),
                                     seq[:, 1:torch.max(seq_len).item()].contiguous().view(-1),
                                     ignore_index=0)
        loss_dict['recon_loss'] = recon_loss
        loss_dict['kl_loss'] = kl_loss
        loss_dict['loss'] = recon_loss + kl_loss
        return loss_dict