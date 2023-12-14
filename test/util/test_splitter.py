import pytest
from moima.utils.splitter.random_splitter import RandomSplitter
from conftest import PipeStorage


@pytest.mark.order(2)
def test_random_splitter():
    datasets = PipeStorage.dataset
    splitter = RandomSplitter(0.8, 0.1, True, 64, 42)
    for key, dataset in datasets.items():
        train, val, test = splitter(dataset)
        batch = next(iter(train))
        assert batch.x.shape[0] == 64
        assert len(train.dataset) == 80
        assert len(val.dataset) == 10
        assert len(test.dataset) == 10
    
    splitter = RandomSplitter(0.8, 0.1, False, 64, 42)
    for key, dataset in datasets.items():
        train, val, test = splitter(dataset)
        batch = next(iter(train))
        assert batch.x.shape[0] == 64
        assert test is None
        assert len(train.dataset) == 80
        assert len(val.dataset) == 10