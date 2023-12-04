import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from moima.pipeline.vade.config import VaDEPipeConfig
from moima.pipeline.pipe import PipeABC
import os
import itertools
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict
import warnings


class VaDEPipe(PipeABC):
    def __init__(self, 
                 config: VaDEPipeConfig,
                 model_state_dict: Dict[str, Any] = None,
                 optimizer_state_dict: Dict[str, Any] = None,
                 **kwargs):
        super().__init__(config, model_state_dict, optimizer_state_dict, **kwargs)
        self.current_epoch = 0
        if self.config.anneal_num_epochs != self.config.num_epochs:
            warnings.warn('The anneal_num_epochs is not equal to num_epochs, '
                            'the anneal_num_epochs will be set to num_epochs.')
            self.config.anneal_num_epochs = self.config.num_epochs
    
    def _forward_batch(self, batch):
        batch.to(self.device)
        self.model.to(self.device)
        x_hat, recon_loss, kl_loss = self.loss_fn(batch, self.model, self.current_epoch)
        self.info = {'Recon loss': recon_loss.item(), 
                     'KL': kl_loss.item(),
                     'gamma_c': f'{self.model.yita_c.min().item()}~{self.model.yita_c.max().item()}'}
        return x_hat, recon_loss + kl_loss
    
    def _interested_info(self, batch, output):
        info = {}
        info.update(self.info)
        info['Label'] = self.featurizer.decode(batch.x[0], is_raw=True)
        info['Reconstruction'] = self.featurizer.decode(output[0], is_raw=False)
        return info
    
    @property
    def custom_saveitems(self):
        return {'featurizer': self.featurizer}
    
    def pretrain(self, pre_epoch=50, retrain=False):
        dataloader = self.train_loader
        pretrained_path = os.path.join(self.config.output_folder, 
                                       f'vade_pretrain_{self.config.desc}.wght')
        
        if (not os.path.exists(pretrained_path)) or retrain==True:
            optimizer = torch.optim.Adam(itertools.chain(
                self.model.encoder.parameters(),\
                self.model.h2mu.parameters(),\
                self.model.h2logvar.parameters(),\
                self.model.decoder.parameters()))

            self.logger.info('Pretrain the VaDE'.center(60,'-'))
            self.model.train()
            n_iter = 0
            for epoch in range(pre_epoch):
                for batch in dataloader:
                    optimizer.zero_grad()
                    batch.to(self.device)
                    seq, seq_len = batch.x, batch.seq_len
                    x_hat, mu, logvar = self.model(batch, is_pretrain=True)
                    recon_loss = F.cross_entropy(x_hat[:, :-1].contiguous().view(-1, x_hat.size(-1)),
                                    seq[:, 1:torch.max(seq_len).item()].contiguous().view(-1),
                                    ignore_index=0)
                    recon_loss = recon_loss*seq.size(1)
                    recon_loss.backward()
                    optimizer.step()
                    n_iter += 1
                    if n_iter % self.config.log_interval == 0:
                        self.logger.info(f'Epoch: {epoch}, Iter: {n_iter}, Rec: {recon_loss.item():.4f}')
            
            self.model.h2logvar.load_state_dict(self.model.h2mu.state_dict())
            self.logger.info('Initialize GMM parameters'.center(60,'-'))
            Z = []
            with torch.no_grad():
                for batch in dataloader:
                    batch.to(self.device)
                    x_hat, mu, logvar = self.model(batch, is_pretrain=True)
                    assert F.mse_loss(mu, logvar) == 0
                    Z.append(mu)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            
            self.model.gmm.fit(Z)
            self.model.pi_.data = torch.from_numpy(self.model.gmm.weights_).to(self.device).float()
            self.model.mu_c.data = torch.from_numpy(self.model.gmm.means_).to(self.device).float()
            self.model.logvar_c.data = torch.log(torch.from_numpy(self.model.gmm.covariances_)).to(self.device).float()

            torch.save(self.model.state_dict(), pretrained_path)
            print(f'Store the pretrain weights at dir {pretrained_path}')

        else:
            self.model.load_state_dict(torch.load(pretrained_path))
    
    def sample(self, num_samples: int=10, center: int=5):
        self.model.eval()
        with torch.no_grad():
            model = self.model.to('cpu')
            featurizer = self.featurizer
            sos_idx = featurizer.charset_dict[featurizer.SOS]
            eos_idx = featurizer.charset_dict[featurizer.EOS]
            pad_idx = featurizer.charset_dict[featurizer.PAD]      
            max_len = featurizer.seq_len
            
            if center is None:
                centers = np.random.choice(self.config.n_clusters, 
                                           size=num_samples, 
                                           p=torch.softmax(self.model.pi_.detach().cpu(), dim=0).numpy())
                mu_c = self.model.mu_c[centers]
                logvar_c = self.model.logvar_c[centers]
                std = torch.exp(0.5 * logvar_c)
                eps = torch.randn_like(std)
                z = mu_c + eps * std
            else:
                mu_c = model.mu_c[center]
                logvar_c = model.logvar_c[center]
                std = torch.exp(0.5 * logvar_c)
                eps = torch.randn_like(std.repeat(num_samples, 1))
                z = mu_c + eps * std
            
            if num_samples == 1:
                z_0 = z.view(1, 1, -1) 
            else:
                z_0 = z.unsqueeze(1)

            h_0 = model.decoder.z2h(z).unsqueeze(0)
            
            w = torch.tensor(sos_idx).repeat(num_samples)
            x = torch.tensor(pad_idx).repeat(num_samples, max_len)
            x[:, 0] = sos_idx
            
            eos_p = torch.tensor(max_len).repeat(num_samples)
            eos_m = torch.zeros(num_samples, dtype=torch.bool)
            
            # sequence part
            for i in range(1, max_len):
                input_emb = model.encoder.embedding(w).unsqueeze(1)
                x_input = torch.cat([input_emb, z_0], dim = -1)
                o, h_0 = model.decoder.gru(x_input, h_0)
                y = model.decoder.output_head(o.squeeze(1))
                y = torch.nn.functional.softmax(y, dim=-1)
                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_m, i] = w[~eos_m]
                eos_mi = ~eos_m & (w == eos_idx)
                eos_mi = eos_mi.type(torch.bool)
                eos_p[eos_mi] = i + 1
                eos_m = eos_m | eos_mi

            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :eos_p[i]])
                
            new_x_cpu = [x.cpu().numpy().tolist() for x in new_x]
        
        smiles_list = []
        for new_x in new_x_cpu:
            smiles = featurizer.decode(new_x + [1], is_raw=True)
            smiles_list.append(smiles)
        return smiles_list
    
    
if __name__ == '__main__':
    config = VaDEPipeConfig.from_args()
    