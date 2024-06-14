import numpy as np
import torch
from rdkit import Chem

from moima.utils.evaluator.generation import GenerationMetrics
from moima.utils.evaluator.regression import RegressionMetrics


def test_generation_metrics():
    sampled_mols = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCC', 'CCCCC', 'TEST']
    train_mols = ['C', 'CC', 'CCC', 'CCCC']
    input_mols = ['C', 'CC', 'CCC', 'CCCC', 'CCCC', 'CCCCCC']
    recon_mols = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC']
    metrics_smiles = GenerationMetrics(sampled_mols, train_mols, input_mols, recon_mols)
    assert metrics_smiles.valid == 7/8
    assert metrics_smiles.unique == 5/7
    assert metrics_smiles.novel == 1/7
    assert metrics_smiles.recon_accuracy == 5/6

    sampled_rdmols = list(map(Chem.MolFromSmiles, sampled_mols))
    train_rdmols = list(map(Chem.MolFromSmiles, train_mols))
    input_rdmols = list(map(Chem.MolFromSmiles, input_mols))
    recon_rdmols = list(map(Chem.MolFromSmiles, recon_mols))
    metrics_rdmols = GenerationMetrics(sampled_rdmols, train_rdmols, input_rdmols, recon_rdmols)
    assert metrics_rdmols.get_metrics() == metrics_smiles.get_metrics()


def test_regression_metrics():
    y_true = torch.FloatTensor([0.1, 0.2, 0.3, 0.4])
    y_pred = torch.FloatTensor([0.11, 0.21, 0.31, 0.41])
    metrics = RegressionMetrics(y_true, y_pred)
    assert np.allclose(metrics.mae, 0.01)
    assert np.allclose(metrics.rmse, 0.01)
    assert np.allclose(metrics.r2, 0.992)
    assert np.allclose(metrics.aard, torch.abs((y_true - y_pred) / y_true).mean().numpy())
    assert np.allclose(metrics.mse, 1e-4)