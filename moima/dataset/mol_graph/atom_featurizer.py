# Generate atom features

from typing import Any, Callable, Dict, List

import numpy as np
import random
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import fsolve
from scipy.spatial import distance_matrix
from collections import defaultdict
from sklearn.decomposition import PCA

Atom = Chem.rdchem.Atom
AtomFeaturesGenerator = Callable[[Atom], np.ndarray]

ATOM_FEATURES_GENERATOR_REGISTRY = {}


class AtomFeaturizer:
    def __init__(self, 
                 generator_name_list: List[str], 
                 params: Dict[str, dict]={}):
        if 'all' in generator_name_list:
            generator_name_list = get_avail_atom_features()
        self.generator_name_list = generator_name_list
        self.params = defaultdict(dict)
        self.params.update(params)
    
    @property
    def available_features(self):
        return get_avail_atom_features()
    
    @property
    def dim(self):
        mol = Chem.MolFromSmiles('C')
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        num_features = self(mol).shape[1]
        return num_features
    
    def __call__(self, mol: Chem.Mol):
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_repr(atom))
        
        for name in self.generator_name_list:
            if 'MOLINPUT' in name:
                add_features = get_atom_feature_generator(name)(mol, **self.params[name])
                atom_features = np.concatenate([atom_features, add_features], axis=1)
        atom_features = torch.from_numpy(np.stack(atom_features, axis=0)).float()
        return atom_features
    
    def get_atom_repr(self, atom: Atom) -> np.ndarray:
        atom_features = []
        for name in self.generator_name_list:
            if 'MOLINPUT' in name:
                continue
            if name not in self.available_features:
                raise ValueError(f'Features generator "{name}" could not be found.')
            if name in self.params:
                atom_features += get_atom_feature_generator(name)(atom, **self.params[name])
            else:
                atom_features += get_atom_feature_generator(name)(atom)
        return np.array(atom_features)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.generator_name_list})'


def one_hot_encoding(value: int, choices: List) -> List:
    r"""Generates a one-hot encoding of the value.
    
    Args:
        value (int): The value to encode.
        choices (List): The considered value choices.
    
    Returns:
        List: A one-hot encoding of the value (length: :obj:`len(choices) + 1`).
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def register_atom_features_generator(features_generator_name: str) \
                                    -> Callable[[AtomFeaturesGenerator], AtomFeaturesGenerator]:
    r"""Decorates a function as a feature generator and registers it in global 
        dictionaries to enable access by name.
    
    Args:
        features_generator_name (str): The name to use to access the features 
            generator.
    
    Returns:
        Callable[[AtomFeaturesGenerator], AtomFeaturesGenerator]: A decorator 
            which will add a atom features generator to the registry using the 
            specified name.
    """
    def decorator(features_generator: AtomFeaturesGenerator) -> AtomFeaturesGenerator:
        ATOM_FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_atom_feature_generator(features_generator_name: str) -> AtomFeaturesGenerator:
    r"""Gets a registered features generator by name.
    
    Args:
        features_generator_name (str): The name of the features generator.
    
    Returns:
        AtomFeaturesGenerator: The desired features generator.
    """
    if features_generator_name not in ATOM_FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found.')

    return ATOM_FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_avail_atom_features() -> List[str]:
    r"""Returns a list of names of available features generators."""
    return list(ATOM_FEATURES_GENERATOR_REGISTRY.keys())


@register_atom_features_generator('atomic_num')
def atom_type_features_generator(atom: Atom, 
                                 choices: List[int]=list(range(100))) -> List:
    r"""Generates a one-hot encoding of the atom type. The default choices are
        the first 100 atomic numbers.
    """
    atom_type_value = atom.GetAtomicNum()
    return one_hot_encoding(atom_type_value, choices)


@register_atom_features_generator('degree')
def degree_features_generator(atom: Atom) -> List:
    r"""Generates a one-hot encoding of the atom degree. The default choices are
        [0, 1, 2, 3, 4].
    """
    degree_choices = list(range(5))
    degree = atom.GetTotalDegree()
    return one_hot_encoding(degree, degree_choices)


@register_atom_features_generator('chiral_tag')
def chiral_tag_features_generator(atom: Atom) -> List:
    r"""Generates a one-hot encoding of the atom chiral tag. The default choices
        are [0, 1, 2, 3].
    """
    chiral_tag_choices = list(range(len(Chem.ChiralType.names)-1))
    chiral_tag = atom.GetChiralTag()
    return one_hot_encoding(chiral_tag, chiral_tag_choices)


@register_atom_features_generator('num_Hs')
def num_Hs_features_generator(atom: Atom) -> List:
    r"""Generates a one-hot encoding of the number of hydrogens. The default
        choices are [0, 1, 2, 3, 4].
    """
    num_Hs_choices = list(range(5))
    num_Hs = atom.GetTotalNumHs()
    return one_hot_encoding(num_Hs, num_Hs_choices)


@register_atom_features_generator('hybridization')
def hybridization_features_generator(atom: Atom) -> List:
    r"""Generates a one-hot encoding of the atom hybridization. The default
        choices are [0, 1, 2, 3, 4, 5, 6, 7, 8].
    """
    hybridization_choices = list(range(len(Chem.HybridizationType.names)-1))
    hybridization = int(atom.GetHybridization())
    return one_hot_encoding(hybridization, hybridization_choices)


@register_atom_features_generator('aromatic')
def aromatic_features_generator(atom: Atom) -> List:
    r"""Generates a one-hot encoding of whether the atom is aromatic."""
    return [1 if atom.GetIsAromatic() else 0]


@register_atom_features_generator('formal_charge')
def formal_charge_features_generator(atom: Atom) -> List:
    r"""Generates a one-hot encoding of the atom formal charge. The default
        choices are [-2, -1, 0, 1, 2].
    """
    formal_charge_choices = [-2, -1, 0, 1, 2]
    formal_charge = atom.GetFormalCharge()
    return one_hot_encoding(formal_charge, formal_charge_choices)


@register_atom_features_generator('mass')
def mass_features_generator(atom: Atom) -> List:
    r"""Generates the atom mass."""
    return [atom.GetMass() * 0.01]


# @register_atom_features_generator('tsne_MOLINPUT')
def tsne_generator(mol: Chem.Mol, 
                   feature_dim: int=3, 
                   perplexity: float=1):
    r"""Generates the t-SNE features of the atom."""
    assert mol.GetNumConformers() > 0, 'No conformer found.'
    pos = mol.GetConformer().GetPositions()
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=feature_dim, perplexity=perplexity)
    pos = tsne.fit_transform(pos)
    return pos


# @register_atom_features_generator('invariant_pca_MOLINPUT')
def invariant_pca_generator(mol: Chem.Mol):
    r"""Generates the invariant PCA features of the atom."""
    assert mol.GetNumConformers() > 0, 'No conformer found.'
    pos = mol.GetConformer().GetPositions()
    if pos.shape[0] == 1:
        return np.zeros((1, 3))
    pca = PCA(n_components=3)
    pca.fit(pos)
    comp = pca.components_
    sign_variants = np.array([[1, 1, 1],
                    [1, 1, -1],
                    [1, -1, 1],
                    [1, -1, -1],
                    [-1, 1, 1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [-1, -1, -1]])
    index = random.randint(0, sign_variants.shape[0] - 1)
    sign = sign_variants[index]
    comp = comp * sign
    pos = np.dot(pos, comp.T)
    return pos


# @register_atom_features_generator('geo_env_MOLINPUT')
def geometry_env_generator(mol: Chem.Mol, 
                           feature_dim: int=32,
                           min_dist: float=1.0,
                           cutoff: float=5.0, 
                           n: int=1):
    r"""Generates the geometry environment of the atom."""
    assert mol.GetNumConformers() > 0, 'No conformer found.'
    pos = mol.GetConformer().GetPositions()
    
    def rbf_expansion(x, y, cutoff=5.0, n=1):
        return np.sqrt(2.0 / cutoff) * np.sin(np.pi * x * n / cutoff) / x - y

    max_y = rbf_expansion(min_dist, 0.0, cutoff, n)
    min_y = rbf_expansion(cutoff, 0.0, cutoff, n)
    y_grid = np.linspace(min_y, max_y, feature_dim + 1)
    dist_ranges = fsolve(rbf_expansion, x0=np.ones(feature_dim + 1), args=(y_grid,))
    dist_ranges = np.flip(dist_ranges).reshape(-1, 1, 1)
    dist_ranges.squeeze()
    
    dist = distance_matrix(pos, pos)
    dist_repeat = np.concatenate([[dist]] * feature_dim, axis=0)
    result = np.logical_and(dist_repeat >= dist_ranges[:-1], 
                            dist_repeat < dist_ranges[1:]).sum(axis=1).T
    return result