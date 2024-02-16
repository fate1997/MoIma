from moima.dataset.mol_graph.data import GraphData
from torch import Tensor
from faenet.model import FAENet as FAENetBase
from faenet.fa_forward import model_forward
import torch
from copy import deepcopy


class FAENet(FAENetBase):
    def __init__(self, 
                 cutoff: float = 6, 
                 act: str = "swish", 
                 preprocess: str = "base_preprocess", 
                 complex_mp: bool = True, 
                 max_num_neighbors: int = 30, 
                 num_gaussians: int = 100, 
                 num_filters: int = 480, 
                 hidden_channels: int = 400, 
                 tag_hidden_channels: int = 0, 
                 pg_hidden_channels: int = 32, 
                 phys_hidden_channels: int = 0, 
                 phys_embeds: bool = False, 
                 num_interactions: int = 5, 
                 mp_type: str = "updownscale_base", 
                 graph_norm: bool = True, 
                 second_layer_MLP: bool = True, 
                 skip_co: str = "False", 
                 energy_head: str = '', 
                 regress_forces: str = '', 
                 force_decoder_type: str  = "mlp"):
        super().__init__(cutoff, act, preprocess, complex_mp, max_num_neighbors, num_gaussians, num_filters, 
                         hidden_channels, tag_hidden_channels, pg_hidden_channels, phys_hidden_channels, 
                         phys_embeds, num_interactions, mp_type, graph_norm, second_layer_MLP, skip_co, 
                         energy_head, regress_forces, force_decoder_type, None)
    
    def forward(self, batch: GraphData) -> Tensor:
        if isinstance(batch, list):
            batch = batch[0]
        if not hasattr(batch, "natoms"):
            batch.natoms = torch.unique(batch.batch, return_counts=True)[1]

        # Distinguish Frame Averaging prediction from traditional case.
        original_pos = batch.pos
        e_all, f_all, gt_all = [], [], []

        # Compute model prediction for each frame
        for i in range(len(batch.fa_pos)):
            batch.pos = batch.fa_pos[i]
            # Forward pass
            preds = super().forward(deepcopy(batch))
            e_all.append(preds["energy"])
            fa_rot = None

            batch.pos = original_pos

            # Average predictions over frames
            preds["energy"] = sum(e_all) / len(e_all)

        if preds["energy"].shape[-1] == 1:
            preds["energy"] = preds["energy"]

        return preds["energy"]