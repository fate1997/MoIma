from moima.model.vae.encoders import GRUEncoder
import pytest
from conftest import PipeStorage
import torch
from moima.model.vae.decoders import GRUDecoder


@pytest.mark.order(2)
def test_gru_encoder_decoder():
    seq_dataset = PipeStorage.dataset['seq']
    batch = seq_dataset.collate_fn([seq_dataset[0], seq_dataset[1]])
    vocab_dim = len(seq_dataset.featurizer.vocab)
    encoder = GRUEncoder(vocab_dim=vocab_dim,
                            emb_dim=128,
                            enc_hidden_dim=292,
                            latent_dim=292,
                            dropout=0.2)
    mu, logvar = encoder(batch)
    assert mu.shape == (2, 292)
    assert logvar.shape == (2, 292)
    assert isinstance(mu, torch.Tensor)
    assert isinstance(logvar, torch.Tensor)
    assert torch.isnan(mu).sum() == 0
    assert torch.isnan(logvar).sum() == 0
    
    decoder = GRUDecoder(embedding=encoder.embedding,
                            emb_dropout=0.2,
                            latent_dim=292,
                            hidden_dim=501,
                            vocab_dim=vocab_dim,
                            emb_dim=128)
    output = decoder(mu, batch)
    assert output.shape[0] == 2
    assert output.shape[1] <= 120
    assert output.shape[2] == vocab_dim
    assert isinstance(output, torch.Tensor)
    assert torch.isnan(output).sum() == 0