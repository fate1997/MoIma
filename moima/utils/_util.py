import logging
import logging.config
import os
import sys

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
                        format='%(asctime)s - [%(levelname)s] - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(log_dir, name)),
                                  logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(name)

    return logger