import torch
import torch.nn.functional as F
import numpy as np


class VaDELossCalc:
    def __init__(self, 
                 start_kl_weight: float = 0.0,
                 end_kl_weight: float = 0.0025,
                 num_epochs: int = 100,
                 n_cycles: int = 5,
                 ratio: float = 0.7,
                 start_center_weight: float = 0.0,
                 end_center_weight: float = 1e7):
        self.kl_scheduler = self.loss_annealing(start_kl_weight, 
                                                end_kl_weight, 
                                                num_epochs, 
                                                n_cycles, 
                                                ratio)
        self.center_scheduler = self.loss_annealing(start_center_weight,
                                                    end_center_weight,
                                                    num_epochs,
                                                    n_cycles,
                                                    ratio)
    
    def loss_annealing(self, 
                       start_weight: float = 0.0,
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
            
    def __call__(self, batch, x_hat, mu, logvar, model, current_epoch):
        seq = batch.x
        seq_len = batch.seq_len
        recon_loss = F.cross_entropy(x_hat[:, :-1].contiguous().view(-1, x_hat.size(-1)),
                                    seq[:, 1:torch.max(seq_len).item()].contiguous().view(-1),
                                    ignore_index=0)
        kl_loss = self.kl_scheduler[current_epoch] * model.gmm_kl_div(mu, logvar)
        center_mut_dists = self.center_scheduler[current_epoch] * model.mus_mutual_distance()
        
        self.loss_items = {'recon_loss': recon_loss.item(),
                           'kl_loss': kl_loss.item(),
                           'center_mut_dists': center_mut_dists.item()}
        
        return recon_loss + kl_loss + center_mut_dists