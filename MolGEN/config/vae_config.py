import pathlib
from dataclasses import dataclass

from MolGEN.config._base import BaseConfig
import torch
import os


PACKAGE_PATH = '../'


@dataclass
class VAEPipeConfig(BaseConfig):
    model: dict
    train: dict
    dataset: dict
    
    @classmethod
    def from_default(cls):
        cls.model = {
            'latent_dim':292,
        }
        cls.train = {
            'lr': 1e-3,
            'num_epochs': 100,
            'batch_size': 128,
        }
        cls.dataset = {
            'path': os.path.join(PACKAGE_PATH, 'example/ZINC', 'zinc_dataset.pt')
        }
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return cls


if __name__ == '__main__':
    vae_config = VAEPipeConfig.from_default()
    print(vae_config.model)
    print(vae_config.train)
    print(vae_config.dataset)
    print(vae_config.device)