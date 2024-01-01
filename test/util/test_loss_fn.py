import pytest
from conftest import PipeStorage
from moima.utils.loss_fn.vae_loss import VAELossCalc
from moima.utils.loss_fn.vade_loss import VaDELossCalc
from moima.utils.loss_fn import build_loss_fn


@pytest.mark.order(3)
def test_vae_loss():
    vae_model = PipeStorage.model['vae']
    loss_fn = VAELossCalc()
    dataset = PipeStorage.dataset['seq']
    splitter = PipeStorage.splitter['random']
    batch = next(iter(splitter(dataset)[0]))
    x_hat, mu, logvar = vae_model(batch)
    loss_dict = loss_fn(batch, mu, logvar, x_hat, 0)
    assert loss_dict.keys() == {'recon_loss', 'kl_loss', 'loss'}
    assert loss_dict['kl_loss'].item() == 0.0
    
    next_loss_dict = loss_fn(batch, mu, logvar, x_hat, 1)
    assert next_loss_dict['kl_loss'].item() != 0.0


@pytest.mark.order(3)
def test_vade_loss():
    vade = PipeStorage.model['vade']
    loss_fn = VaDELossCalc()
    dataset = PipeStorage.dataset['seq']
    splitter = PipeStorage.splitter['random']
    batch = next(iter(splitter(dataset)[0]))
    x_hat, mu, logvar, eta_c = vade(batch)
    loss_dict = loss_fn(batch, mu, logvar, x_hat, 
                        eta_c, vade.pi_, vade.mu_c, vade.logvar_c)
    assert loss_dict.keys() == {'recon_loss', 'kl_loss', 'loss'}


@pytest.mark.order(3)
def test_loss_fn_factory():
    vae_loss_fn = build_loss_fn(name='vae_loss', ratio=0.7)
    assert isinstance(vae_loss_fn, VAELossCalc)
    vade_loss_fn = build_loss_fn(name='vade_loss', kl_weight=1)
    assert isinstance(vade_loss_fn, VaDELossCalc)


@pytest.mark.order(3)
def test_regression_loss():
    loss_names = ['mse', 'l1', 'huber']
    dataset = PipeStorage.dataset['desc']
    splitter = PipeStorage.splitter['random']
    batch = next(iter(splitter(dataset)[0]))
    mlp = PipeStorage.model['mlp']
    y_pred = mlp(batch)
    for name in loss_names:
        loss_args = {'name':name}
        loss_fn = build_loss_fn(**loss_args)
        loss = loss_fn(batch.y, y_pred)
        assert loss.item() >= 0.0