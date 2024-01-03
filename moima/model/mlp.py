from torch import nn, Tensor

from moima.model._util import get_activation, init_weight
from moima.dataset.descriptor_vec.data import VecBatch


class MLP(nn.Module):
    r"""Multi-layer perceptron.
    
    Args:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        output_dim (int): Output dimension.
        n_layers (int): Number of layers.
        dropout (float): Dropout rate.
        activation (str): Activation function.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 output_dim: int = 1, 
                 n_layers: int = 2, 
                 dropout: float = 0.2, 
                 activation: str = 'relu'):
        super().__init__()
        
        activation = get_activation(activation)
        
        if n_layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            self.layers = []
            for i in range(n_layers - 1):
                self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
                self.layers.append(activation)
                self.layers.append(nn.LayerNorm(hidden_dim))
                self.layers.append(nn.Dropout(dropout))

            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.Sequential(*self.layers)
        
        self.layers.apply(init_weight)
        
    def forward(self, batch) -> Tensor:
        r"""Forward pass of :class:`MLP`.

        Args:
            batch (VecBatch): Batch of data. The batch should contain :obj:`x`.         
        """
        x = batch.x
        output = self.layers(x)
        return output
