import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from moima.pipeline.pipe import PipeABC
from moima.pipeline.config import create_config_class
from moima.dataset._abc import DataABC, FeaturizerABC
from moima.utils.evaluator.generation import GenerationMetrics

from typing import Any, Dict


VAEPipeConfig = create_config_class('VAEPipeConfig',
                                    'selfies_seq',
                                    'chemical_vae',
                                    'random',
                                    'vae_loss',
                                    'none')

class VAEPipe(PipeABC):
    r"""Variational autoencoder pipeline.
    
    Args:
        config (VAEPipeConfig): Configuration of the pipeline.
        model_state_dict (Dict[str, Any]): State dictionary of the model. Default to None.
        optimizer_state_dict (Dict[str, Any]): State dictionary of the optimizer. Default to None.
    """
    def __init__(self, 
                 config: VAEPipeConfig, 
                 featurizer: FeaturizerABC = None,
                 model_state_dict: Dict[str, Any] = None,
                 optimizer_state_dict: Dict[str, Any] = None,
                 scheduler_state_dict: Dict[str, Any] = None,
                 is_training: bool = True):
        super().__init__(config, 
                         featurizer,
                         model_state_dict, 
                         optimizer_state_dict, 
                         scheduler_state_dict,
                         is_training)
    
    def _forward_batch(self, batch, calc_loss=True):
        r"""Forward a batch of data."""
        batch.to(self.device)
        self.model.to(self.device)
        x_hat, mu, logvar = self.model(batch)
        loss = {}
        if calc_loss:
            loss = self.loss_fn(batch, mu, logvar, x_hat, self.current_epoch) 
        
        info = {}
        info['Label'] = self.featurizer.decode(batch.x[0], is_raw=True)
        info['Reconstruction'] = self.featurizer.decode(x_hat[0], is_raw=False)
        self.interested_info.update(info)
        return x_hat, loss
    
    def set_interested_info(self):
        return super().set_interested_info()
    
    def eval(self, loader_name: str='test'):
        self.logger.info('Evaluating'.center(60, "-"))
        loader = self.loader['train']
        train_smiles = self.batch_flatten(loader, 
                                          register_items=['smiles'],
                                          register_output=False)['smiles']
        loader = self.loader[loader_name]
        eval_outputs = self.batch_flatten(loader, register_items=['smiles'])
        eval_smiles = eval_outputs['smiles']
        eval_recon_smiles = []
        if isinstance(eval_outputs['output'], Tensor):
            eval_recon_smiles = [self.featurizer.decode(x, is_raw=False) \
                                 for x in eval_outputs['output']]
        else:
            for batch_x in eval_outputs['output']:
                for x in batch_x:
                    eval_recon_smiles.append(self.featurizer.decode(x, is_raw=False))
        sampled_smiles = self.sample(10000)
        metrics = GenerationMetrics(sampled_smiles,
                                    train_smiles,
                                    eval_smiles,
                                    eval_recon_smiles)
        return metrics.get_metrics()
        
    def sample(self, num_samples: int=10, label: int=None):
        self.model.eval()
        with torch.no_grad():
            model = self.model.to('cpu')
            featurizer = self.featurizer
            sos_idx = featurizer.vocab_dict[featurizer.SOS]
            eos_idx = featurizer.vocab_dict[featurizer.EOS]
            pad_idx = featurizer.vocab_dict[featurizer.PAD]      
            max_len = featurizer.seq_len
            
            z = torch.randn(num_samples, model.encoder.h2mu.out_features)
            if model.consider_label:
                if label is None:
                    label = torch.randint(0, model.num_classes, (num_samples,))
                else:
                    label = torch.tensor(label).repeat(num_samples)
                z = model.conditional_z(z, label)
            if num_samples == 1:
                z_0 = z.view(1, 1, -1) 
            else:
                z_0 = z.unsqueeze(1)

            h_0 = model.decoder.z2h(z).repeat(self.model.decoder.num_layers, 1, 1)
            
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
            smiles = featurizer.decode(torch.Tensor(new_x + [eos_idx]), is_raw=True)
            smiles_list.append(smiles)
        return smiles_list
    
if __name__ == '__main__':
    config = VAEPipeConfig.from_args()
    