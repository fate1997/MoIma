import os

from moima.dataset.descriptor_vec.dataset import DescDataset, VecBatch, VecData
from moima.dataset.smiles_seq.data import SeqBatch, SeqData
from moima.dataset.smiles_seq.dataset import SeqDataset


def test_abc_and_seq_dataset(zinc1k, helpers):
    
    dataset = SeqDataset(zinc1k, 
                        featurizer_kwargs={'seq_len': 120},
                        vocab_path=None,
                        processed_path=None,
                        force_reload=False,
                        save_processed=False)
    assert len(dataset) == 1000
    assert isinstance(dataset[0], SeqData)
    
    zinc1k_processed = os.path.splitext(zinc1k)[0] + '.pt'
    zinc1k_vocab = os.path.splitext(zinc1k)[0] + '_vocab.pkl'
    helpers.remove_files(zinc1k_processed, zinc1k_vocab)
    dataset = SeqDataset(zinc1k,
                        featurizer_kwargs={'seq_len': 120},
                        vocab_path=None,
                        processed_path=None,
                        force_reload=False,
                        save_processed=False)
    assert not os.path.exists(zinc1k_processed)
    assert os.path.exists(zinc1k_vocab)
    
    dataset = SeqDataset(zinc1k,
                        featurizer_kwargs={'seq_len': 120},
                        vocab_path=None,
                        processed_path=None,
                        force_reload=False,
                        save_processed=True)
    
    assert os.path.exists(zinc1k_processed)
    
    dataset = SeqDataset(zinc1k,
                        featurizer_kwargs={'seq_len': 120},
                        vocab_path=None,
                        processed_path=zinc1k_processed,
                        force_reload=False,
                        save_processed=False)

    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert isinstance(batch, SeqBatch)
    assert batch.x.shape == (2, 120)
    assert batch.seq_len.shape == (2, )
    assert batch.smiles == [dataset[0].smiles, dataset[1].smiles]
    
    dataset = SeqDataset(zinc1k,
                        featurizer_kwargs={'seq_len': 120},
                        additional_cols=['logP', 'qed'],
                        vocab_path=None,
                        processed_path=None,
                        force_reload=True,
                        save_processed=False)
    print(dataset[0].__dict__)
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
    
    helpers.remove_files(zinc1k_processed, zinc1k_vocab)


def test_desc_dataset(zinc1k):
    dataset = DescDataset(zinc1k,
                          label_col='logP',
                          additional_cols=['qed', 'SAS'],
                          featurizer_kwargs={'mol_desc': 'ecfp,rdkit',
                                             'ecfp_radius': 4,
                                             'ecfp_n_bits': 2048},
                          force_reload=False,
                          save_processed=False)
    assert len(dataset) == 1000
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