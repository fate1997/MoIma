from typing import Callable, Union
from torch_geometric.nn.models import DimeNetPlusPlus as DimeNetPlusPlusPyG
from moima.dataset.mol_graph.data import GraphData
from torch import Tensor


class DimeNetPP(DimeNetPlusPlusPyG):
    def __init__(self, 
                 hidden_channels: int=128,
                out_channels: int=1,
                num_blocks: int=3,
                int_emb_size: int=64,
                basis_emb_size: int=8,
                out_emb_channels: int=256,
                num_spherical: int=7,
                num_radial: int=6,
                cutoff: float = 5.0,
                max_num_neighbors: int = 32,
                envelope_exponent: int = 5,
                num_before_skip: int = 1,
                num_after_skip: int = 2,
                num_output_layers: int = 3,
                act: str = 'swish',
                output_initializer: str = 'zeros'):
        super().__init__(hidden_channels, out_channels, num_blocks, int_emb_size, basis_emb_size,
                        out_emb_channels, num_spherical, num_radial, cutoff, max_num_neighbors , envelope_exponent, num_before_skip, num_after_skip, num_output_layers, act, output_initializer)
        
    def forward(self, data: GraphData) -> Tensor:
        z, pos, batch = data.z, data.pos, data.batch
        return super().forward(z, pos, batch)
