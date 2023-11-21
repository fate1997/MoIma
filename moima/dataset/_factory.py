from ._abc import DatasetABC
from .smiles_seq.dataset import SMILESSeq
from typing import List, Literal


DATASET_REGISTRY = {
    "smiles_seq": SMILESSeq,
}


class DatasetFactory:
    """Factory class for dataset."""
        
    @staticmethod
    def create(name: Literal["smiles_seq"], **kwargs) -> DatasetABC:
        """Create a dataset instance by name.
        
        Args:
            name (str): Name of the dataset.
            **kwargs: Arguments for the dataset.
        
        Returns:
            A dataset instance.
        """
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Dataset {name} is not available.")
        return DATASET_REGISTRY[name](**kwargs)
    
    @property
    def avail(self) -> List[str]:
        """List of available annotate."""
        return list(DATASET_REGISTRY.keys())