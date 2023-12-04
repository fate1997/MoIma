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