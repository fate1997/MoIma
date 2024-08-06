from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor

from moima.dataset._abc import DataABC


class GraphData(Data, DataABC):
    def __init__(self, 
                 x: Optional[Tensor] = None, 
                 edge_index: OptTensor = None, 
                 edge_attr: OptTensor = None, 
                 y: OptTensor = None, 
                 pos: OptTensor = None, 
                 smiles: Optional[str] = None,
                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.smiles = smiles