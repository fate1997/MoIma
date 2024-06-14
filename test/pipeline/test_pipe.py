import os

import pytest
import numpy as np
from torch.utils.data import DataLoader

from moima.pipeline.downstream_pipe import DownstreamPipe, create_downstream_config_class
from moima.pipeline.vade_pipe import VaDEPipe, VaDEPipeConfig
from moima.pipeline.vae_pipe import VAEPipe, VAEPipeConfig


@pytest.fixture(scope='session')
def vae_pipe(zinc1k):
    config = VAEPipeConfig(raw_path=zinc1k, 
                            desc='pytest_run', 
                            seq_len=120,
                            save_interval=50000,
                            log_interval=5,
                            vocab_size=100,
                            num_epochs=50)
    pipe = VAEPipe(config)
    return pipe


@pytest.mark.order(4)
def test_pipe_attr(vae_pipe):
    assert hasattr(vae_pipe, 'config')
    assert hasattr(vae_pipe, 'device')
    assert hasattr(vae_pipe, 'n_epoch')
    assert hasattr(vae_pipe, 'logger')
    assert hasattr(vae_pipe, 'workspace')
    assert os.path.exists(vae_pipe.workspace)
    assert os.path.exists(os.path.join(vae_pipe.workspace, 'pipe.log'))
    assert hasattr(vae_pipe, 'featurizer')
    assert vae_pipe.featurizer.__class__.__name__ == 'SeqFeaturizer'
    assert hasattr(vae_pipe, 'model')
    assert vae_pipe.model.__class__.__name__ == 'ChemicalVAE'
    assert hasattr(vae_pipe, 'loss_fn')
    assert vae_pipe.loss_fn.__class__.__name__ == 'VAELossCalc'
    assert hasattr(vae_pipe, 'optimizer')
    assert hasattr(vae_pipe, 'loader')
    assert isinstance(vae_pipe.loader, dict)
    assert isinstance(vae_pipe.loader['train'], DataLoader)
    
    assert vae_pipe.config.vocab_size == vae_pipe.featurizer.vocab_size


@pytest.mark.order(4)
def test_build_module(vae_pipe):
    dataset = vae_pipe.build_dataset()
    assert dataset.__class__.__name__ == 'SeqDataset'
    
    loader = vae_pipe.build_loader()
    assert isinstance(loader, dict)
    assert isinstance(loader['train'], DataLoader)
    
    model = vae_pipe.build_model()
    assert model.__class__.__name__ == 'ChemicalVAE'
    assert next(model.parameters()).device.type == vae_pipe.device


@pytest.mark.order(4)
def test_io(vae_pipe):
    pipe_path = vae_pipe.save()
    assert os.path.exists(pipe_path)
    loaded_pipe = VAEPipe.from_pretrained(pipe_path)
    assert vae_pipe.config == loaded_pipe.config
    assert vae_pipe.featurizer.vocab == loaded_pipe.featurizer.vocab
    for name, param in vae_pipe.model.named_parameters():
        assert param.data.equal(loaded_pipe.model.state_dict()[name])
    
    pytest.pretrained_path = pipe_path
        
        
@pytest.mark.order(4)
def test_train(vae_pipe):
    vae_pipe.train()
    loss = []
    for _, items in vae_pipe.training_trace.items():
        print(items)
        loss.append(items['loss'])
    assert loss[-1] - loss[0] < 0


@pytest.mark.order(4)
def test_vade(zinc1k):
    config = VaDEPipeConfig(raw_path=zinc1k, 
                        desc='pytest_run',
                        save_interval=5000,
                        log_interval=1000,
                        num_epochs=1000,
                        batch_size=64,
                        latent_dim=256,
                        n_clusters=10)
    pipe = VaDEPipe(config)
    
    # Test `_forward_batch` function
    batch = next(iter(pipe.loader['train']))
    x_hat, loss_dict = pipe._forward_batch(batch)
    assert x_hat.shape[0] == config.batch_size
    assert x_hat.shape[-1] == config.vocab_size
    assert 'loss' in loss_dict.keys()
    
    # Test `pretrain` function
    pretrained_path = pipe.pretrain(pre_epoch=3, retrain=True)
    assert os.path.exists(pretrained_path)

    # Test `sample` function
    sampled_smiles = pipe.sample(10)
    assert len(sampled_smiles) == 10
    assert isinstance(sampled_smiles[0], str)


@pytest.mark.order(5)
def test_downstream(zinc1k):
    DownstreamPipeConfig = create_downstream_config_class(
                                        "DownstreamPipeConfig",
                                        dataset_name="desc_vec",
                                        model_name='mlp',
                                        splitter_name='random',
                                        scheduler_name='none',
                                        loss_fn_name='mse')
    
    config = DownstreamPipeConfig(
        raw_path=zinc1k,
        input_dim=128,
        desc='pytest_run',
        activation='leaky_relu',
        log_interval=100,
        num_epochs=500,
        lr=1e-3,
        dataset_name='desc_vec',
        label_col="logP",
        mol_desc='addi_dict',
        pretrained_pipe_class='VAEPipe',
        pretrained_pipe_path=pytest.pretrained_path)
    pipe = DownstreamPipe(config)
    
    # Test `desc_from_pretrained` function
    desc_dict = pipe._desc_from_pretrained()
    assert isinstance(desc_dict, dict)
    assert len(desc_dict) == 1000
    assert isinstance(list(desc_dict.values())[0], np.ndarray)
    assert isinstance(list(desc_dict.keys())[0], str)
    
    pipe.train(1)