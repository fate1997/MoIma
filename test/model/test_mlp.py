from moima.model.mlp import MLP
import pytest
from conftest import PipeStorage
import torch


@pytest.mark.order(2)
def test_mlp():
    dataset = PipeStorage.dataset['desc']
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    input_dim = batch.x.shape[1]
    model = MLP(input_dim=input_dim, 
                hidden_dim=200, 
                output_dim=1, 
                n_layers=2, 
                dropout=0.1, 
                activation='relu')
    output = model(batch)
    assert output.shape == (2, 1)
    assert isinstance(output, torch.Tensor)
    assert torch.isnan(output).sum() == 0
    
    PipeStorage.model['mlp'] = model