from typing import List, Literal

from torch import nn

from ._abc import LossCalcABC
from .vade_loss import VaDELossCalc
from .vae_loss import VAELossCalc
from .regression_loss import MSELossCalc, L1LossCalc, HuberLossCalc

LOSS_FN_REGISTRY = {
    "vae_loss": VAELossCalc,
    "vade_loss": VaDELossCalc,
    "mse": MSELossCalc,
    "l1": L1LossCalc,
    "huber": HuberLossCalc,
}


def build_loss_fn(name: str=None, **kwargs) -> LossCalcABC:
    """Build a loss calculator instance by name.
    
    Args:
        name (str): Name of the loss calculator. If None, the name will be read from kwargs.
        **kwargs: Arguments for the loss calculator.
    
    Returns:
        A loss calculator instance.
    """
    if name is None:
        name = kwargs.pop('name')
    if name not in LOSS_FN_REGISTRY:
        raise ValueError(f"Loss calcualtor {name} is not available.")
    return LOSS_FN_REGISTRY[name](**kwargs)