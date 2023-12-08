import os
import random

import pandas as pd
import pytest

from moima.dataset.smiles_seq.data import SeqData
from moima.dataset.smiles_seq.featurizer import SeqFeaturizer


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


def test_reload_vocab(zinc1k):
    df = pd.read_csv(zinc1k)
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