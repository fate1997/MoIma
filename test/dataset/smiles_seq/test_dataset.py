import os

from moima.dataset.smiles_seq.data import SeqData, SeqBatch
from moima.dataset.smiles_seq.dataset import SeqDataset


def test_dataset(zinc1k, helpers):
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
    
    helpers.remove_files(zinc1k_processed, zinc1k_vocab)

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
                        force_reload=False,
                        save_processed=False)
    assert hasattr(dataset[0], 'logP')
    assert hasattr(dataset[0], 'qed')
    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert hasattr(batch, 'logP')
    assert hasattr(batch, 'qed')
    assert batch.logP.shape == (2, )
    assert batch.qed.shape == (2, )