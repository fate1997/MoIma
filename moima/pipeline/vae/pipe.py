import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from moima.pipeline.vae.config import VAEPipeConfig
from moima.pipeline._abc import PipeABC


class VAEPipe(PipeABC):
    def __init__(self, config: VAEPipeConfig):
        super().__init__(config)
    
    def _forward_batch(self, batch):
        batch.to(self.device)
        mu, logvar, rec_loss, x_hat = self.model(batch.x, batch.seq_len)
        loss = self.loss_fn(mu, logvar, rec_loss, self.current_epoch)
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
    