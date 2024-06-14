from moima.dataset.mol_graph.data import GraphData
from torch import Tensor
from faenet.model import FAENet as FAENetBase
from faenet.fa_forward import model_forward
import torch
from copy import deepcopy
from torch import nn
from egnn_pytorch import EGNN_Sparse
from torch_geometric.nn import radius_graph

def nan_to_num(vec, num=0.0):
    idx = torch.isnan(vec)
    vec[idx] = num
    return vec

def _normalize(vec, dim=-1):
    return nan_to_num(
        torch.div(vec, torch.norm(vec, dim=dim, keepdim=True)))


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
                 force_decoder_type: str  = "mlp",
                 trainable_pca: bool=False,
                 num_atom_features: int = 85,):
        super().__init__(cutoff, act, preprocess, complex_mp, max_num_neighbors, num_gaussians, num_filters, 
                         hidden_channels, tag_hidden_channels, pg_hidden_channels, phys_hidden_channels, 
                         phys_embeds, num_interactions, mp_type, graph_norm, second_layer_MLP, skip_co, 
                         energy_head, regress_forces, force_decoder_type, None)
        self.trainable_pca = trainable_pca
        if self.trainable_pca and False:
            print("Using trainable PCA")
            self.embdding = nn.Embedding(85, 2) # 128 -> 2
            self.embdding.reset_parameters()
        if self.trainable_pca:
            print("Using trainable PCA")
            self.egnn_layer = EGNN_Sparse(num_atom_features)
            self.feat_proj = nn.Linear(num_atom_features, 2, bias=False)
            nn.init.xavier_uniform_(self.feat_proj.weight)
    
    def forward(self, batch: GraphData) -> Tensor:
        if isinstance(batch, list):
            batch = batch[0]
        if not hasattr(batch, "natoms"):
            batch.natoms = torch.unique(batch.batch, return_counts=True)[1]

        # Distinguish Frame Averaging prediction from traditional case.
        original_pos = batch.pos
        e_all, f_all, gt_all = [], [], []
        
        # Trainable PCA
        if self.trainable_pca:
            """
            atomic_emb = self.embdding(batch.z)
            """
            input_feats = torch.cat([batch.pos, batch.x], dim=-1)
            edge_index = radius_graph(
                        batch.pos,
                        r=5,
                        batch=batch.batch)
            output_feats = self.egnn_layer(input_feats, edge_index)
            pos, feats = output_feats[:, :3], output_feats[:, 3:]
            atomic_emb = self.feat_proj(feats)
            pos = pos - pos.mean(dim=0, keepdim=True) # new added
            pos_emb = torch.matmul(atomic_emb.t(), pos) # [2, 3]
            # pos_emb = pos_emb - pos_emb.mean(dim=0, keepdim=True)
            # build node-wise frame
            pos1, pos2 = pos_emb[0], pos_emb[1]
            node_diff = pos1 - pos2
            node_diff = _normalize(node_diff)
            node_cross = torch.cross(pos1, pos2)
            node_cross = _normalize(node_cross)
            node_vertical = torch.cross(node_diff, node_cross)
            # node_frame shape: (num_nodes, 3, 3)
            node_frame = torch.cat((node_diff.unsqueeze(-1), node_cross.unsqueeze(-1), node_vertical.unsqueeze(-1)), dim=-1)
            eigenvec = node_frame
            # # Eigendecomposition
            # C = torch.matmul(pos_emb.t(), pos_emb)
            # eigenval, eigenvec = torch.linalg.eigh(C)
            self.eigenvec = eigenvec
            fa_pos = original_pos @ eigenvec
            batch.fa_pos = [fa_pos]
            
        # Compute model prediction for each frame
        for i in range(len(batch.fa_pos)):
            batch.pos = batch.fa_pos[i]
            # Forward pass
            preds = super().forward(batch)
            e_all.append(preds["energy"])
            fa_rot = None

            batch.pos = original_pos

            # Average predictions over frames
            preds["energy"] = sum(e_all) / len(e_all)

        if preds["energy"].shape[-1] == 1:
            preds["energy"] = preds["energy"]

        return preds["energy"]