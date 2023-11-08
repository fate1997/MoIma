import torch
from torch import nn
from MolGEN.model import ChemicalVAE
from torch.utils.data import DataLoader
from MolGEN.data_pipeline.dataset import SMILESDataset
from MolGEN.data_pipeline.featurizer import SMILESFeaturizer
from MolGEN.config.vae_config import VAEPipeConfig
from tqdm import tqdm


class VAEPipe:
    def __init__(self, config: VAEPipeConfig):
        self.config = config
        self.model = ChemicalVAE(**config.model).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=config.train['lr'])
        self.dataset = SMILESDataset(raw_path=config.dataset['raw_path'],
                                     Featurizer=SMILESFeaturizer)
        self.featurizer = self.dataset.featurizer
        self.train_loader = DataLoader(self.dataset, 
                                        batch_size=config.train['batch_size'], 
                                        shuffle=True)
    
    def train(self):
        self.model.train()
        for epoch in range(self.config.train['num_epochs']):
            loss_sum = 0
            num_examples = 0
            for i, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
                if i == 0:
                    print(f"Input SMILES: {self.featurizer.decode(batch[0])}")
                self.optimizer.zero_grad()
                x = batch.to(self.config.device)
                x_hat, mu, log_var = self.model(x)
                loss = self.model.loss_func(x, x_hat, mu, log_var)
                loss.backward()
                self.optimizer.step()
                
                if i == 0:
                    print(f"Output SMILES: {self.featurizer.decode(x_hat[0])}")
                
                loss_sum += loss.item() * x.size(0)
                num_examples += x.size(0)
            loss_avg = loss_sum / num_examples
            if epoch % 1 == 0:
                print(f"Epoch: {epoch}, Loss: {loss_avg}")
                torch.save(self.model.state_dict(), self.config.train['model_path'])
    
    
if __name__ == '__main__':
    config = VAEPipeConfig.from_default()
    config.dataset['raw_path'] = 'example/ZINC/zinc250k.csv'
    config.train['model_path'] = 'example/ZINC/vae_model.pt'
    pipe = VAEPipe(config)
    pipe.train()