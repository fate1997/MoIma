import pickle
import warnings
from copy import deepcopy
import numpy as np
from typing import List, Dict
from scipy.sparse import coo_matrix

import torch
from rdkit import Chem
from tqdm import tqdm

from moima.dataset._abc import FeaturizerABC
from moima.dataset.mol_graph.data import GraphData
from moima.dataset.mol_graph.atom_featurizer import AtomFeaturizer
from moima.dataset.mol_graph.bond_featurizer import BondFeaturizer
from moima.typing import MolRepr


class GraphFeaturizer(FeaturizerABC):
        
    def __init__(self, 
                 atom_feature_names: List[str],
                 bond_feature_names: List[str]=[],
                 atom_feature_params: Dict[str, dict]={},
                 assign_pos: bool = False):
        self.atom_feature_names = atom_feature_names
        self.bond_feature_names = bond_feature_names
        self.atom_featurizer = AtomFeaturizer(atom_feature_names, 
                                              atom_feature_params)
        self.bond_featurizer = BondFeaturizer(bond_feature_names)
        self.assign_pos = assign_pos
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.atom_feature_names}, {self.bond_feature_names})'
    
    def cal_pos(self, mol: Chem.Mol) -> torch.Tensor:
        if mol.GetNumConformers() == 0:
            return
        else:
            return mol.GetConformer().GetPositions()
    
    def encode(self, mol: MolRepr) -> GraphData:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return None
        smiles = Chem.MolToSmiles(mol)
        
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.atom_featurizer(atom))
        atom_features = torch.from_numpy(np.stack(atom_features, axis=0)).float()
        
        # Edge index
        adj = Chem.GetAdjacencyMatrix(mol)
        coo_adj = coo_matrix(adj)
        edge_index = torch.from_numpy(np.vstack([coo_adj.row, coo_adj.col])).long()
        
        # Bond features
        bond_features = []
        for bond in mol.GetBonds():
            bond_features.append(self.bond_featurizer(bond))
        bond_features = torch.from_numpy(np.stack(bond_features, axis=0))
        
        # Get positions
        if self.assign_pos:
            pos = self.cal_pos(mol)
        else:
            pos = None
        
        # Construct the graph data
        graph_data = GraphData(x=atom_features, 
                               edge_index=edge_index, 
                               edge_attr=bond_features,
                               smiles=smiles,
                               pos=pos)
        return graph_data

    @property
    def arg4model(self) -> dict:
        return {'num_atom_features': self.atom_featurizer.dim}