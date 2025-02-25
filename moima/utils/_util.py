import logging
import logging.config
import os
import sys
import random

import numpy as np
import torch


def KL_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Calculate the KL divergence between the prior and the posterior.
    
    Args:
        mu (torch.Tensor): The mean of the posterior.
        logvar (torch.Tensor): The log variance of the posterior.
    
    Returns:
        torch.Tensor: The KL divergence.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def get_logger(name: str, log_dir: str):
    """Get a logger."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(log_dir, name)),
                                  logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(name)

    return logger


class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""

    def __init__(self, 
                 patience: int=100, 
                 save_func: callable=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_func = save_func
        self.last_save_path = None

    def __call__(self, val_metric: float):

        score = -val_metric

        if self.best_score is None:
            self.best_score = score
            self.last_save_path = self.save_func(verbose=False)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.last_save_path = self.save_func(verbose=False)
            self.counter = 0

def set_random_seed(seed: int=42) -> None:
    """Set random seed globally."""
    # python and numpy
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch cpu
    torch.manual_seed(seed)
    # torch gpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False