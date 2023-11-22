from .vae.chemical_vae import ChemicalVAE
from torch import nn
from typing import Literal, List
from dataclasses import Field


MODEL_REGISTRY = {
    "chemical_vae": ChemicalVAE,
}


class ModelFactory:
    """Factory class for model."""
        
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
    
    @property
    def avail(self) -> List[str]:
        """List of available models."""
        return list(MODEL_REGISTRY.keys())