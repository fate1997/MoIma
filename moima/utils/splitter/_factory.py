from typing import List, Literal

from ._abc import SplitterABC
from .random_splitter import RandomSplitter

SPLITTER_REGISTRY = {
    "random": RandomSplitter,
}


class SplitterFactory:
    """Factory class for splitter."""
        
    @staticmethod
    def create(**kwargs) -> SplitterABC:
        """Create a splitter instance by name.
        
        Args:
            name (str): Name of the splitter.
            **kwargs: Arguments for the splitter.
        
        Returns:
            A splitter instance.
        """
        name = kwargs.pop('name')
        if name not in SPLITTER_REGISTRY:
            raise ValueError(f"Dataset {name} is not available.")
        return SPLITTER_REGISTRY[name](**kwargs)
    
    @property
    def avail(self) -> List[str]:
        """List of available splitters."""
        return list(SPLITTER_REGISTRY.keys())