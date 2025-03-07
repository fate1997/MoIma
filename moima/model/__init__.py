from dataclasses import Field
from typing import List

from torch import nn

from .gnn.dimenet import DimeNet
from .gnn.dimenet_pp import DimeNetPP
from .gnn.faenet import FAENet
from .gnn.gat_v2 import GATv2
from .gnn.gat_v2_salt import GATv2_Salt
from .gnn.gatv2_pyg import GATv2_PyG
from .mlp import MLP
from .vae.chemical_vae import ChemicalVAE
from .vae.vade import VaDE

MODEL_REGISTRY = {
    "chemical_vae": ChemicalVAE,
    "vade": VaDE,
    "mlp": MLP,
    "gat_v2": GATv2,
    "dimenet": DimeNet,
    "dimenet++": DimeNetPP,
    "faenet": FAENet,
    "gatv2_salt": GATv2_Salt,
    "gatv2_pyg": GATv2_PyG,
}


def build_model(name: str=None, **kwargs) -> nn.Module:
    """Build a model instance by name.
    
    Args:
        name (str): Name of the model. If None, the name will be read from kwargs.
        **kwargs: Arguments for the model.
    
    Returns:
        A model instance.
    """
    if name is None:
        name = kwargs.pop('name')
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Dataset {name} is not available.")
    return MODEL_REGISTRY[name](**kwargs)


class ModelFactory:
    """Factory class for model."""
    avail = list(MODEL_REGISTRY.keys())
        
    @staticmethod
    def create(**kwargs) -> nn.Module:
        """Create a model instance by name.
        
        Args:
            name (str): Name of the model.
            **kwargs: Arguments for the model.
        
        Returns:
            A model instance.
        """
        name = kwargs.pop('name')
        if isinstance(name, Field):
            name = name.default
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Dataset {name} is not available.")
        return MODEL_REGISTRY[name](**kwargs)