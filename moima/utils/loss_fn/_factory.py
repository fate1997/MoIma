from typing import List, Literal

from ._abc import LossCalcABC
from .vae_loss import VAELossCalc
from .vade_loss import VaDELossCalc

LOSS_FN_REGISTRY = {
    "vae_loss": VAELossCalc,
    "vade_loss": VaDELossCalc,
}


class LossCalcFactory:
    """Factory class for loss calculator."""
        
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
    
    @property
    def avail(self) -> List[str]:
        """List of available splitters."""
        return list(LOSS_FN_REGISTRY.keys())