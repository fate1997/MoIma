import torch
from torch import nn
from torch.nn import functional as F


class ChemicalVAE(nn.Module):
    def __init__(self, 
                 input_dim=120,
                 vocab_size=35,
                 latent_dim=292):
        super().__init__()
        
        self.encoder = Encoder(input_dim, vocab_size, latent_dim)
        self.decoder = Decoder(latent_dim, vocab_size, input_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def loss_func(self, 
                  x: torch.Tensor, x_hat: torch.Tensor, 
                  mu: torch.Tensor, logvar: torch.Tensor):
        reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence


class Encoder(nn.Module):
    def __init__(self, 
                 input_dim=120,
                 channels=35,
                 output_dim=292):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 9, kernel_size=9)
        self.conv2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv3 = nn.Conv1d(9, 10, kernel_size=11)
        
        self.selu = nn.SELU()
        
        self.linear0 = nn.Linear((channels - 26) * 10, 435)
        self.mean_head = nn.Linear(435, output_dim)
        self.std_head = nn.Linear(435, output_dim)
    
    def forward(self, x: torch.Tensor):
        x = self.selu(self.conv1(x))
        x = self.selu(self.conv2(x))
        x = self.selu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        x = self.selu(self.linear0(x))
        return self.mean_head(x), self.std_head(x)


class Decoder(nn.Module):
    def __init__(self, 
                 input_dim=292,
                 channels=35,
                 output_dim=120):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, input_dim)
        self.selu = nn.SELU()
        self.gru = nn.GRU(input_dim, 501, 3, batch_first=True)
        self.output_head = nn.Linear(501, channels)
        self.softmax = nn.Softmax(dim=1)
        self.output_dim = output_dim
    
    def forward(self, z: torch.Tensor):
        z = self.selu(self.linear(z))
        z = z.view(z.size(0), 1, z.size(-1))
        z = z.repeat(1, self.output_dim, 1)
        z, _ = self.gru(z)
        z = z.contiguous().view(-1, z.size(-1))
        z = self.output_head(z)
        z = self.softmax(z)
        z = z.contiguous().view(-1, self.output_dim, z.size(-1))
        return z


if __name__ == '__main__':
    # Test Encoder
    encoder = Encoder(channels=36)
    x = torch.randn(5, 120, 36).softmax(dim=2)
    output = encoder(x)
    print(f"Encoder output shape: {output[0].shape}")
    
    # Test Decoder
    decoder = Decoder(channels=36)
    z = torch.randn(5, 292)
    output = decoder(z)
    print(f"Decoder output shape: {output.shape}")
    
    # Test ChemicalVAE
    model = ChemicalVAE(vocab_size=36)
    output = model(x)
    print(f"ChemicalVAE output shape: {output[0].shape}")
    # print(output)
    loss = model.loss_func(x, output[0], output[1], output[2])
    # print(f"Loss: {loss}")
    
    