from moima.dataset import DatasetFactory
from moima.dataset.smiles_seq.dataset import SeqDataset
from moima.dataset.descriptor_vec.dataset import DescDataset


def test_dataset_factory(zinc100):
    assert DatasetFactory.avail == ['smiles_seq', 'desc_vec']
    seq_dataset = DatasetFactory.create(name='smiles_seq',
                                        raw_path=zinc100,
                                    featurizer_kwargs={'seq_len': 120})
    assert isinstance(seq_dataset, SeqDataset)
    assert len(seq_dataset) == 100
    
    desc_dataset = DatasetFactory.create(name='desc_vec',
                                         raw_path=zinc100,
                                         label_col='logP',
                                        featurizer_kwargs={'mol_desc': 'ecfp',
                                                        'ecfp_radius': 4,
                                                        'ecfp_n_bits': 2048})
    assert isinstance(desc_dataset, DescDataset)
    assert len(desc_dataset) == 100
    assert desc_dataset[0].y.shape == (1, )
    assert desc_dataset[0].x.shape == (2048, )
    