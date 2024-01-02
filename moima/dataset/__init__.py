from typing import List

from ._abc import DatasetABC, FeaturizerABC
from .descriptor_vec.dataset import DescDataset, DescFeaturizer
from .smiles_seq.dataset import SeqDataset, SeqFeaturizer
from .mol_graph.dataset import GraphDataset, GraphFeaturizer

DATASET_REGISTRY = {
    "smiles_seq": SeqDataset,
    "desc_vec": DescDataset,
    'graph': GraphDataset,
}

FEATURIZER_REGISTRY = {
    "smiles_seq": SeqFeaturizer,
    "desc_vec": DescFeaturizer,
    'graph': GraphFeaturizer,
}


def build_dataset(name: str=None, **kwargs) -> DatasetABC:
    """Build a dataset instance by name.
    
    Args:
        name (str): Name of the dataset. If None, the name will be read from kwargs.
        **kwargs: Arguments for the dataset.
    
    Returns:
        A dataset instance.
    """
    if name is None:
        name = kwargs.pop('name')
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} is not available.")
    return DATASET_REGISTRY[name](**kwargs)


def build_featurizer(name: str=None, **kwargs) -> FeaturizerABC:
    """Build a featurizer instance by name.
    
    Args:
        name (str): Name of the featurizer. If None, the name will be read from kwargs.
        **kwargs: Arguments for the featurizer.
    
    Returns:
        A featurizer instance.
    """
    if name is None:
        name = kwargs.pop('name')
    if name not in FEATURIZER_REGISTRY:
        raise ValueError(f"Featurizer {name} is not available.")
    return FEATURIZER_REGISTRY[name](**kwargs)