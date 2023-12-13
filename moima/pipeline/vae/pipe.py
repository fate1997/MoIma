import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from moima.pipeline.vae.config import VAEPipeConfig
from moima.pipeline.pipe import PipeABC

from typing import Any, Dict


class VAEPipe(PipeABC):
    def __init__(self, config: VAEPipeConfig, 
                 model_state_dict: Dict[str, Any] = None,
                 optimizer_state_dict: Dict[str, Any] = None,
                 **kwargs):
        super().__init__(config, model_state_dict, optimizer_state_dict, **kwargs)
    
    def _forward_batch(self, batch):
        batch.to(self.device)
        mu, logvar, x_hat = self.model(batch)
        loss = self.loss_fn(batch, mu, logvar, x_hat, self.current_epoch)
        return x_hat, loss
    
    def _interested_info(self, batch, output):
        info = {}
        info['Label'] = self.featurizer.decode(batch.x[0], is_raw=True)
        info['Reconstruction'] = self.featurizer.decode(output[0], is_raw=False)
        return info
    
    @property
    def custom_saveitems(self):
        return {'featurizer': self.featurizer}
    
    
    
    
if __name__ == '__main__':
    config = VAEPipeConfig.from_args()
    