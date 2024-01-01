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