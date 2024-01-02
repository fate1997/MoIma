import os

import pytest
from conftest import PipeStorage

from moima.dataset import build_dataset
from moima.dataset.descriptor_vec.dataset import DescDataset, VecBatch, VecData, DescFeaturizer
from moima.dataset.smiles_seq.data import SeqBatch, SeqData
from moima.dataset.smiles_seq.dataset import SeqDataset, SeqFeaturizer
from moima.dataset.mol_graph.dataset import GraphDataset, GraphFeaturizer, GraphData
from torch_geometric.data import Batch


@pytest.fixture(scope='session')
def seq_featurizer():
    return SeqFeaturizer(seq_len=120)


@pytest.fixture(scope='session')
def desc_featurizer():
    return DescFeaturizer(mol_desc='ecfp,rdkit',
                          ecfp_radius=4,
                          ecfp_n_bits=2048)


@pytest.mark.order(1)
def test_abc_and_seq_dataset(zinc100, helpers, seq_featurizer):
    featurizer = SeqFeaturizer(seq_len=120)
    dataset = SeqDataset(zinc100, 
                        featurizer=seq_featurizer,
                        vocab_path=None,
                        processed_path=None,
                        force_reload=False,
                        save_processed=False)
    assert len(dataset) == 100
    assert isinstance(dataset[0], SeqData)
    
    zinc100_processed = os.path.splitext(zinc100)[0] + '.pt'
    zinc100_vocab = os.path.splitext(zinc100)[0] + '_vocab.pkl'
    helpers.remove_files(zinc100_processed, zinc100_vocab)
    dataset = SeqDataset(zinc100,
                        featurizer=seq_featurizer,
                        vocab_path=None,
                        processed_path=None,
                        force_reload=False,
                        save_processed=False)
    assert not os.path.exists(zinc100_processed)
    assert os.path.exists(zinc100_vocab)
    
    dataset = SeqDataset(zinc100,
                        featurizer=seq_featurizer,
                        vocab_path=None,
                        processed_path=None,
                        force_reload=False,
                        save_processed=True)
    
    assert os.path.exists(zinc100_processed)
    
    dataset = SeqDataset(zinc100,
                        featurizer=seq_featurizer,
                        vocab_path=None,
                        processed_path=zinc100_processed,
                        force_reload=False,
                        save_processed=False)

    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert isinstance(batch, SeqBatch)
    assert batch.x.shape == (2, 120)
    assert batch.seq_len.shape == (2, )
    assert batch.smiles == [dataset[0].smiles, dataset[1].smiles]
    
    dataset = SeqDataset(zinc100,
                        featurizer=seq_featurizer,
                        additional_cols=['logP', 'qed'],
                        vocab_path=None,
                        processed_path=None,
                        force_reload=True,
                        save_processed=False)

    assert hasattr(dataset[0], 'logP')
    assert hasattr(dataset[0], 'qed')
    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert hasattr(batch, 'logP')
    assert hasattr(batch, 'qed')
    assert batch.logP.shape == (2, )
    assert batch.qed.shape == (2, )
    
    assert isinstance(batch[0], SeqData)
    assert hasattr(batch[0], 'logP')
    assert hasattr(batch[0], 'qed')
    
    helpers.remove_files(zinc100_processed, zinc100_vocab)
    
    PipeStorage.dataset['seq'] = dataset


@pytest.mark.order(1)
def test_desc_dataset(zinc100, desc_featurizer):
    dataset = DescDataset(zinc100,
                          label_col='logP',
                          additional_cols=['qed', 'SAS'],
                          featurizer=desc_featurizer,
                          force_reload=False,
                          save_processed=False)
    assert len(dataset) == 100
    assert isinstance(dataset[0], VecData)
    assert dataset[0].y.shape == (1, )
    assert hasattr(dataset[0], 'qed')
    assert hasattr(dataset[0], 'SAS')
    assert len(dataset.featurizer.columns) == dataset[0].x.shape[0]
    assert dataset.featurizer.columns[0] == 'ecfp_0'
    
    vec_batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert isinstance(vec_batch, VecBatch)
    assert vec_batch.x.shape == (2, len(dataset.featurizer.columns))
    assert vec_batch.y.shape == (2, 1)
    assert vec_batch.qed.shape == (2, )
    assert vec_batch.SAS.shape == (2, )
    assert vec_batch.smiles == [dataset[0].smiles, dataset[1].smiles]
    
    PipeStorage.dataset['desc'] = dataset


@pytest.mark.order(1)
def test_graph_dataset(zinc100):
    featurizer = GraphFeaturizer(atom_feature_names=['atomic_num'],
                             bond_feature_names=['bond_type'],
                             assign_pos=False)
    dataset = GraphDataset(raw_path=zinc100,
                        label_col=['logP'],
                        featurizer=featurizer)
    assert len(dataset) == 100
    assert isinstance(dataset[0], GraphData)
    assert dataset[0].x.shape == (dataset[0].num_nodes, featurizer.atom_featurizer.dim)
    assert dataset[0].edge_index.shape == (2, dataset[0].num_edges)
    assert dataset[0].edge_attr is not None
    assert dataset[0].pos is None
    assert dataset[0].y.shape == (1, )
    
    batch = next(iter(dataset.create_loader(batch_size=24)))
    assert isinstance(batch, Batch)
    assert batch.batch is not None
    assert batch.batch.max() == 23
    

def test_dataset_factory(zinc100, desc_featurizer, seq_featurizer):
    seq_dataset = build_dataset(name='smiles_seq',
                                raw_path=zinc100,
                                featurizer=seq_featurizer)
    assert isinstance(seq_dataset, SeqDataset)
    assert len(seq_dataset) == 100
    
    kwargs = {'name': 'smiles_seq',
              'raw_path': zinc100,
              'featurizer': seq_featurizer}
    seq_dataset = build_dataset(**kwargs)
    assert isinstance(seq_dataset, SeqDataset)
    assert len(seq_dataset) == 100
    
    kwargs = {'raw_path': zinc100,
              'label_col': 'logP',
              'featurizer': desc_featurizer}
    desc_dataset = build_dataset(name='desc_vec', **kwargs)
    assert isinstance(desc_dataset, DescDataset)
    assert len(desc_dataset) == 100
    assert desc_dataset[0].y.shape == (1, )
    assert desc_dataset[0].x.shape == (2259, )