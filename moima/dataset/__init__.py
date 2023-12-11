from typing import List

from ._abc import DatasetABC
from .descriptor_vec.dataset import DescDataset
from .smiles_seq.dataset import SeqDataset

DATASET_REGISTRY = {
    "smiles_seq": SeqDataset,
    "desc_vec": DescDataset,
}


class DatasetFactory:
    """Factory class for dataset."""
        
    @staticmethod
    def create(**kwargs) -> DatasetABC:
        """Create a dataset instance by name.
        
        Args:
            name (str): Name of the dataset.
            **kwargs: Arguments for the dataset.
        
        Returns:
            A dataset instance.
        """
        name = kwargs.pop('name')
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Dataset {name} is not available.")
        return DATASET_REGISTRY[name](**kwargs)
    
    @property
    def avail(self) -> List[str]:
        """List of available datasets."""
        return list(DATASET_REGISTRY.keys())