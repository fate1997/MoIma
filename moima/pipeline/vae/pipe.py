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
    
    def sample(self, num_samples: int=10):
        self.model.eval()
        with torch.no_grad():
            model = self.model.to('cpu')
            featurizer = self.featurizer
            sos_idx = featurizer.charset_dict[featurizer.SOS]
            eos_idx = featurizer.charset_dict[featurizer.EOS]
            pad_idx = featurizer.charset_dict[featurizer.PAD]      
            max_len = featurizer.seq_len
            
            z = torch.randn(num_samples, model.h2mu.out_features)
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
    config = VAEPipeConfig.from_args()
    