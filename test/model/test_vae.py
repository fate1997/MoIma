import pytest
from conftest import PipeStorage

from moima.model.vae.chemical_vae import ChemicalVAE
from moima.model.vae.vade import VaDE
import torch


@pytest.mark.order(2)
def test_chemicalvae():
    dataset = PipeStorage.dataset['seq']
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    vocab_size = len(dataset.featurizer.vocab)
    model = ChemicalVAE(vocab_size=vocab_size,
                        enc_hidden_dim=128,
                        latent_dim=64,
                        emb_dim=256,
                        dec_hidden_dim=100,
                        dropout=0.1)
    output = model(batch)
    assert isinstance(output, tuple)
    assert isinstance(output[0], torch.Tensor)
    assert isinstance(output[1], torch.Tensor)
    assert isinstance(output[2], torch.Tensor)
    assert torch.isnan(output[0]).sum() == 0
    assert torch.isnan(output[1]).sum() == 0
    assert torch.isnan(output[2]).sum() == 0
    assert output[0].size() == (2, 120, vocab_size)
    assert output[1].size() == (2, 64)
    assert output[2].size() == (2, 64)
    
    mu = model.get_repr(batch)
    assert isinstance(mu, torch.Tensor)
    assert torch.isnan(mu).sum() == 0
    assert mu.size() == (2, 64)
    
    PipeStorage.model['vae'] = model


@pytest.mark.order(2)
def test_vade():
    dataset = PipeStorage.dataset['seq']
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    vocab_size = len(dataset.featurizer.vocab)
    model = VaDE(vocab_size=vocab_size,
                 enc_hidden_dim=128,
                 latent_dim=64,
                 emb_dim=256,
                 dec_hidden_dim=100,
                 n_clusters=10,
                 dropout=0.1)
    output = model(batch)
    assert isinstance(output, tuple)
    assert isinstance(output[0], torch.Tensor)
    assert isinstance(output[1], torch.Tensor)
    assert isinstance(output[2], torch.Tensor)
    assert isinstance(output[3], torch.Tensor)
    assert torch.isnan(output[0]).sum() == 0
    assert torch.isnan(output[1]).sum() == 0
    assert torch.isnan(output[2]).sum() == 0
    assert torch.isnan(output[3]).sum() == 0
    assert output[0].size() == (2, 120, vocab_size)
    assert output[1].size() == (2, 64)
    assert output[2].size() == (2, 64)
    assert output[3].size() == (2, 10)
    
    latent_repr = model.get_repr(batch)
    assert isinstance(latent_repr, torch.Tensor)
    assert torch.isnan(latent_repr).sum() == 0
    assert latent_repr.size() == (2, 64)
    
    guassian_pdf_log = model.gaussian_pdfs_log(latent_repr, model.mu_c, model.logvar_c)
    assert isinstance(guassian_pdf_log, torch.Tensor)
    assert torch.isnan(guassian_pdf_log).sum() == 0
    assert guassian_pdf_log.size() == (2, 10)
    
    PipeStorage.model['vade'] = model