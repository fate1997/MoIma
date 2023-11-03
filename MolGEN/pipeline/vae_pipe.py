import torch
from torch import nn
from MolGEN.model import ChemicalVAE
from torch.utils.data import DataLoader
from MolGEN.config.vae import VAEPipeConfig
from tqdm import tqdm


class VAEPipe:
    def __init__(self, config: VAEPipeConfig):
        self.config = config
        self.model = ChemicalVAE(**config.model).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=config.train['lr'])
        self.dataset = torch.load(config.dataset['path'])
        self.train_loader = DataLoader(self.dataset['train'], 
                                        batch_size=config.train['batch_size'], 
                                        shuffle=True)
    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.config['train']['num_epochs']), desc='Training'):
            loss_sum = 0
            num_examples = 0
            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x = batch.to(self.config['device'])
                x_hat, mu, log_var = self.model(x)
                loss = self.model.loss_func(x, x_hat, mu, log_var)
                loss.backward()
                self.optimizer.step()
                
                loss_sum += loss.item() * x.size(0)
                num_examples += x.size(0)
            loss_avg = loss_sum / num_examples
            if epoch % 1 == 0:
                print(f"Epoch: {epoch}, Loss: {loss_avg.item()}")
    
    
if __name__ == '__main__':
    config = VAEPipeConfig.from_default()
    config.dataset['path'] = 'example/ZINC/zinc250k.pt'
    pipe = VAEPipe(config)
    pipe.train()