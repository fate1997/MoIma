from typing import List, Literal

from torch import nn

from ._abc import LossCalcABC
from .vade_loss import VaDELossCalc
from .vae_loss import VAELossCalc

LOSS_FN_REGISTRY = {
    "vae_loss": VAELossCalc,
    "vade_loss": VaDELossCalc,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "huber": nn.SmoothL1Loss,
}


class LossCalcFactory:
    """Factory class for loss calculator."""
    avail = list(LOSS_FN_REGISTRY.keys())
        
    @staticmethod
    def create(**kwargs) -> LossCalcABC:
        """Create a loss calculator instance by name.
        
        Args:
            name (str): Name of the loss calculator.
            **kwargs: Arguments for the loss calculator.
        
        Returns:
            A loss calculator instance.
        """
        name = kwargs.pop('name')
        if name not in LOSS_FN_REGISTRY:
            raise ValueError(f"Loss calcualtor {name} is not available.")
        return LOSS_FN_REGISTRY[name](**kwargs)