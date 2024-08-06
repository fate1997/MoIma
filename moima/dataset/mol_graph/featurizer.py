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
from moima.dataset.mol_graph.transform import get_transform
from moima.typing import MolRepr


class GraphFeaturizer(FeaturizerABC):
        
    def __init__(self, 
                 atom_feature_names: List[str],
                 bond_feature_names: List[str]=[],
                 atom_feature_params: Dict[str, dict]={},
                 assign_pos: bool = False,
                 transform_names: List[str]=[],):
        self.atom_feature_names = atom_feature_names
        self.bond_feature_names = bond_feature_names
        self.atom_featurizer = AtomFeaturizer(atom_feature_names, 
                                              atom_feature_params)
        if len(bond_feature_names) > 0:
            self.bond_featurizer = BondFeaturizer(bond_feature_names)
        else:
            self.bond_featurizer = None
        self.assign_pos = assign_pos
        
        if len(transform_names) > 0:
            self.transforms = [get_transform(name) for name in transform_names]
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.atom_feature_names}, {self.bond_feature_names})'
    
    def cal_pos(self, mol: Chem.Mol) -> torch.Tensor:
        if mol.GetNumConformers() == 0:
            return
        else:
            pos = mol.GetConformer().GetPositions()
            from sklearn.decomposition import PCA
            import random
            pos = pos - np.mean(pos, axis=0)
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
            # return torch.as_tensor(pos).float()
            return torch.as_tensor(mol.GetConformer().GetPositions()).float()
    
    def encode(self, mol: MolRepr) -> GraphData:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return None
        smiles = Chem.MolToSmiles(mol)
        
        # Atom features
        atom_features = self.atom_featurizer(mol)
        z = []
        for atom in mol.GetAtoms():
            z.append(atom.GetAtomicNum())
        z = torch.tensor(z, dtype=torch.long)
        
        # Edge index
        adj = Chem.GetAdjacencyMatrix(mol)
        coo_adj = coo_matrix(adj)
        edge_index = torch.from_numpy(np.vstack([coo_adj.row, coo_adj.col])).long()
        
        # Bond features
        if self.bond_featurizer is None:
            bond_features = None
        else:
            bond_features = []
            for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                    bond_features.append(self.bond_featurizer(mol.GetBondBetweenAtoms(i, j)))
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
                               pos=pos,
                               z=z)
        
        # Apply the transform
        if hasattr(self, 'transforms'):
            for transform in self.transforms:
                graph_data = transform(graph_data)
        
        if smiles.count('.') == 1:
            comp1, comp2 = Chem.GetMolFrags(mol)
            node_comp = torch.zeros(graph_data.num_nodes)
            node_comp[torch.LongTensor(comp2)] = 1
            edge_comp = torch.zeros(graph_data.num_edges)
            comp2_tensor = torch.LongTensor([comp2])
            edge_mask = (edge_index[0].repeat(len(comp2), 1).T == comp2_tensor).any(dim=1)
            edge_comp[edge_mask] = 1
            graph_data.node_comp = node_comp.long()
            graph_data.edge_comp = edge_comp.long()
            
        return graph_data

    @property
    def arg4model(self) -> dict:
        return {'num_atom_features': self.atom_featurizer.dim}