from typing import Any
import torch
from ._abc import LossCalcABC
import numpy as np
import torch.nn.functional as F
from moima.utils._util import KL_divergence
from moima.dataset._abc import DataABC


class VAELossCalc(LossCalcABC):
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
                 batch: DataABC,
                 mu: torch.Tensor,
                 logvar: torch.Tensor,
                 x_hat: torch.Tensor,
                 current_epoch: int) -> torch.Tensor:
        """Calculate the loss."""
        kl_weight = self.kl_scheduler[current_epoch]
        kl_loss = kl_weight * KL_divergence(mu, logvar) / mu.size(0)
        seq = batch.x
        seq_len = batch.seq_len
        recon_loss = F.cross_entropy(x_hat[:, :-1].contiguous().view(-1, x_hat.size(-1)),
                                              seq[:, 1:torch.max(seq_len).item()].contiguous().view(-1),
                                              ignore_index=0)
        return recon_loss + kl_loss