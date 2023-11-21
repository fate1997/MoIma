import argparse
from dataclasses import dataclass

import torch


@dataclass
class VAEPipeConfig: 
    # Basic
    desc: str = 'debug'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset
    raw_data_path: str = None
    processed_data_path: str = None
    data_replace: bool = False
    
    # Model
    latent_dim: int = 128
    vocab_size: int = 36
    dec_hidden_dim: int = 384
    emb_dim: int = 36
    enc_hidden_dim: int = 256
    
    # Train
    lr: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 128
    output_folder: str = 'output/vae_pipe'
    log_interval: int = 1000
    
    def __repr__(self) -> str:
        string = self.__class__.__name__ + '\n'
        for k, v in self.__dict__.items():
            string += f'{k}: {v}\n'
        return string    
    
    @property
    def model_args(self):
        return {'latent_dim': self.latent_dim,
                'vocab_size': self.vocab_size}
    
    @classmethod
    def from_args(self):
        parser =  argparse.ArgumentParser(description='Parser For Arguments', 
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for k, v in self.__dataclass_fields__.items():
            parser.add_argument(f'-{k}', type=v.type, default=v.default)
        args = parser.parse_args()
        return self(**vars(args))


if __name__ == '__main__':
    vae_config = VAEPipeConfig.from_args()
    print(vae_config)