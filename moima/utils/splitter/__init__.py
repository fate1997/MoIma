from typing import List, Literal

from ._abc import SplitterABC
from .random_splitter import RandomSplitter

SPLITTER_REGISTRY = {
    "random": RandomSplitter,
}


def build_splitter(name: str=None, **kwargs) -> SplitterABC:
    """Build a splitter instance by name.
    
    Args:
        name (str): Name of the splitter. If None, the name will be read from kwargs.
        **kwargs: Arguments for the splitter.
    
    Returns:
        A splitter instance.
    """
    if name is None:
        name = kwargs.pop('name')
    if name not in SPLITTER_REGISTRY:
        raise ValueError(f"Dataset {name} is not available.")
    return SPLITTER_REGISTRY[name](**kwargs)