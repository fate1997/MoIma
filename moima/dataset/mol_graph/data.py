from typing import Optional
from torch import Tensor
from torch_geometric.typing import OptTensor
from moima.dataset._abc import DataABC
from torch_geometric.data import Data


class GraphData(Data, DataABC):
    def __init__(self, 
                 x: Tensor | None = None, 
                 edge_index: OptTensor = None, 
                 edge_attr: OptTensor = None, 
                 y: OptTensor = None, 
                 pos: OptTensor = None, 
                 smiles: Optional[str] = None,
                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.smiles = smiles