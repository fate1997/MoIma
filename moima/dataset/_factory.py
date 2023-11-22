from ._abc import DatasetABC
from .smiles_seq.dataset import SMILESSeq, SeqFeaturizer
from typing import List, Literal


DATASET_REGISTRY = {
    "smiles_seq": (SMILESSeq, SeqFeaturizer)
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
        return DATASET_REGISTRY[name][0](**kwargs)
    
    @property
    def avail(self) -> List[str]:
        """List of available datasets."""
        return list(DATASET_REGISTRY.keys())


class FeaturizerFactory:
    """Factory class for featurizer."""
    
    @staticmethod
    def create(**kwargs):
        """Create a featurizer instance by name.
        
        Args:
            name (str): Name of the featurizer.
            **kwargs: Arguments for the featurizer.
        
        Returns:
            A featurizer instance.
        """
        name = kwargs.pop('name')
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Featurizer {name} is not available.")
        return DATASET_REGISTRY[name][1](**kwargs)
    
    @property
    def avail(self) -> List[str]:
        """List of available featurizers."""
        return list(DATASET_REGISTRY.keys())