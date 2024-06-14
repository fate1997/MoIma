import os
import random

import numpy as np
import pandas as pd
import torch
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from moima.dataset.descriptor_vec.featurizer import (DescFeaturizer, VecData,
                                                     _get_desc_from_dict,
                                                     _get_dict_from_csv,
                                                     _get_ecfp,
                                                     _get_rdkit_desc)
from moima.dataset.mol_graph.atom_featurizer import AtomFeaturizer
from moima.dataset.mol_graph.bond_featurizer import BondFeaturizer
from moima.dataset.smiles_seq.data import SeqData
from moima.dataset.smiles_seq.featurizer import SeqFeaturizer
from moima.dataset.mol_graph.featurizer import GraphFeaturizer, GraphData


@pytest.mark.parametrize("SMILES", ["CC(=O)OC1=CC=CC=C1C(=O)O", 
                                    "CCO", 
                                    "CC(=O)O"])
def test_encode_decode(SMILES):
    featurizer = SeqFeaturizer(seq_len=10)
    seq_data = featurizer.encode(SMILES)
    assert isinstance(seq_data, SeqData)
    assert seq_data.x.size() == (10, )
    
    rebuild_smiles = featurizer.decode(seq_data.x, is_raw=True)
    if len(SMILES) >= 8:
        assert rebuild_smiles == SMILES[:8]
    else:
        assert rebuild_smiles == SMILES


def test_vocab_io():
    random_vocab = [str(chr(random.randrange(0, 20, 1))) for _ in range(20)]
    random_vocab = list(set(random_vocab))
    featurizer = SeqFeaturizer(random_vocab, seq_len=10)
    path = os.path.join(pytest.TEMP_PATH, "test_vocab.pkl")
    save_path = featurizer.save_vocab(path)
    assert save_path == path
    loaded_vocab = featurizer.load_vocab(path)
    assert loaded_vocab == random_vocab
    assert featurizer.vocab == random_vocab


def test_reload_vocab(zinc100):
    df = pd.read_csv(zinc100)
    featurizer = SeqFeaturizer(vocab=['a'], seq_len=120)
    featurizer.reload_vocab(df.smiles)
    vocab = featurizer.vocab
    for smiles in df.smiles:
        smiles = featurizer._double2single(smiles)
        for token in smiles:
            assert token in vocab
    
    additional_dict = {'logP': df.logP, 'qed': df.qed, 'smiles': df.smiles}
    data_list = featurizer(df.smiles, **additional_dict)
    
    assert len(data_list) == len(df)
    assert isinstance(data_list[0], SeqData)
    assert data_list[0].x.size() == (120, )
    assert data_list[0].logP == df.logP[0]
    assert data_list[0].qed == df.qed[0]


def test_ecfp(smiles_batch):
    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        ecfp = _get_ecfp(mol, 4, 2048)
        assert ecfp.shape == (2048, )
        assert type(ecfp) == np.ndarray


def test_rdkit_and_csv(smiles_batch, helpers):
    features = []
    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        rdkit_desc = _get_rdkit_desc(mol)
        assert type(rdkit_desc) == np.ndarray
        features.append(rdkit_desc)
    
    features = np.stack(features)
    columns = ['smiles'] + [x[0] for x in Descriptors._descList]
    assert features.shape == (len(smiles_batch), len(columns)-1)
    
    csv_path = os.path.join(pytest.TEMP_PATH, 'rdkit_desc.csv')
    pd.DataFrame(np.concatenate([np.array(smiles_batch)[:, None], features], axis=1), 
                 columns=columns).to_csv(csv_path, index=False)
    
    assert os.path.exists(csv_path)
    desc_dict = _get_dict_from_csv(csv_path)
    assert type(desc_dict) == dict
    assert len(desc_dict) == len(smiles_batch)

    for i, smiles in enumerate(smiles_batch):
        assert smiles in desc_dict
        assert desc_dict[smiles].shape == features[i].shape
        assert np.allclose(desc_dict[smiles], 
                           features[i])
    
    fake_smiles = 'C'
    data = _get_desc_from_dict(fake_smiles, desc_dict)
    assert data is None

    featurizer = DescFeaturizer('ecfp,rdkit,csv,addi_dict', 
                                desc_csv_path=csv_path,
                                addi_desc_dict=desc_dict)
    output = featurizer.encode(smiles_batch[0])
    assert isinstance(output, VecData)
    assert len(featurizer.columns) == output.x.shape[0]
    desc_length = len(list(desc_dict.values())[0])
    assert output.x.shape == ((desc_length)*3+2048, )
    
    data_list = featurizer(smiles_batch)
    assert len(data_list) == len(smiles_batch)
    assert isinstance(data_list[0], VecData)
    assert len(data_list[0].x) == len(featurizer.columns)
    
    helpers.remove_files(csv_path)


def test_atom_featurizer():
    mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')
    atom = mol.GetAtomWithIdx(0)
    
    names = ['atomic_num', 'degree', 'formal_charge', 'chiral_tag', 
             'hybridization', 'aromatic', 'num_Hs']
    for name in names:
        featurizer = AtomFeaturizer([name])
        atom_features = featurizer(mol)
        assert isinstance(atom_features, torch.Tensor)
        assert atom_features.ndim == 2
        
    featurizer = AtomFeaturizer(['atomic_num'], {'atomic_num': {'choices': [1, 6, 7]}})
    atom_features = featurizer(mol)
    assert atom_features.size(1) == 4
    
    featurizer = AtomFeaturizer(names)
    atom_features = featurizer(mol)
    assert isinstance(atom_features, torch.Tensor)
    assert atom_features.ndim == 2


def test_bond_featurizer():
    mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')
    bond = mol.GetBondWithIdx(0)
    
    names = ['bond_type', 'bond_is_conjugated', 'bond_is_in_ring']
    for name in names:
        featurizer = BondFeaturizer([name])
        bond_features = featurizer(bond)
        assert isinstance(bond_features, np.ndarray)
        assert bond_features.ndim == 1
    
    featurizer = BondFeaturizer(names)
    bond_features = featurizer(bond)
    assert isinstance(bond_features, np.ndarray)
    assert bond_features.ndim == 1


def test_graph_featurizer(smiles_batch):
    featurizer = GraphFeaturizer(['atomic_num', 'degree', 'formal_charge', 'chiral_tag',
                                    'hybridization', 'aromatic', 'num_Hs'],
                                     ['bond_type', 'bond_is_conjugated', 'bond_is_in_ring'],
                                     {'atomic_num': {'choices': [1, 6, 7]}})
    data_list = featurizer(smiles_batch)
    assert len(data_list) == len(smiles_batch)
    assert isinstance(data_list[0], GraphData)
    assert data_list[0].edge_index.shape[0] == 2
    assert data_list[0].x.shape[1] == featurizer.atom_featurizer.dim
    assert data_list[0].edge_attr is not None
    assert data_list[0].pos is None