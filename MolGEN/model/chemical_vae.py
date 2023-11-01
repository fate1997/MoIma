import torch
from torch import nn


class ChemicalVAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


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
        
        self.linear0 = nn.Linear((channels-26)*10, 435)
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
        z = z.contiguous().view(z.size(0), -1, z.size(-1))
        return z