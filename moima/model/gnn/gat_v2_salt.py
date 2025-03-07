from typing import List, Tuple

import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import spherical_harmonics
from mace.modules.blocks import RadialEmbeddingBlock
from torch_geometric.nn import global_add_pool, global_max_pool

from moima.dataset.mol_graph.data import GraphData

from .._util import MLP, get_activation, init_weight


def seed_all():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)


class GATv2_Salt(nn.Module):
    def __init__(
        self, 
        num_atom_features: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_heads: int, 
        pred_hidden_dim: int=128, 
        pred_dropout: float=0.2, 
        pred_layers:int=2,
        activation: str='prelu', 
        residual: bool = True, 
        num_tasks: int = 1,
        bias: bool = True, 
        dropout: float = 0.1
    ):
        super(GATv2_Salt, self).__init__()

        # update phase
        feature_per_layer = [num_atom_features] + [hidden_dim] * num_layers
        layers = []
        for i in range(num_layers):
            layer = GATLayer(
                num_node_features=feature_per_layer[i] * (1 if i == 0 else num_heads),
                output_dim=feature_per_layer[i + 1],
                num_heads=num_heads,
                concat=True if i < len(feature_per_layer) - 2 else False,
                activation=get_activation(activation),
                residual=residual,
                bias=bias,
                dropout=dropout
            )
            layers.append(layer)
        self.gat = nn.Sequential(*layers)

        # readout phase
        self.atom_weighting = nn.Sequential(
            nn.Linear(feature_per_layer[-1], 1),
            nn.Sigmoid()
        )
        self.atom_weighting.apply(init_weight)

        # prediction phase
        self.predict = MLP(
            input_dim=feature_per_layer[-1] * 2 * 2,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )        
        
    def forward(self, batch: GraphData): 
        output, _,= self.gat((batch.x, batch.edge_index))
        batch_index = batch.batch
        mask = batch.node_comp.bool()
        batch_index[mask] = len(batch) + batch_index[mask]
        weighted = self.atom_weighting(output)
        output1 = global_max_pool(output, batch_index)
        output2 = global_add_pool(weighted * output, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        output = torch.cat([output[:len(output)//2], output[len(output)//2:]], dim=1)
        return self.predict(output)


class GATLayer(nn.Module):
    def __init__(self, num_node_features: int, output_dim: int, num_heads: int,
                 activation=nn.PReLU(), concat: bool = True, residual: bool = True,
                 bias: bool = True, dropout: float = 0.1, share_weights: bool = True):
        super(GATLayer, self).__init__()

        seed_all()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.residual = residual
        self.activation = activation
        self.concat = concat
        self.dropout = dropout
        self.share_weights = share_weights

        # Embedding by linear projection
        self.linear_src = nn.Linear(num_node_features, output_dim * num_heads, bias=False)
        if self.share_weights:
            self.linear_dst = self.linear_src
        else:
            self.linear_dst = nn.Linear(num_node_features, output_dim * num_heads, bias=False)


        # The learnable parameters to compute attention coefficients
        self.double_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))

        # Bias and concat
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim * num_heads))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        if residual:
            if num_node_features == num_heads * output_dim:
                self.residual_linear = nn.Identity()
            else:
                self.residual_linear = nn.Linear(num_node_features, num_heads * output_dim, bias=False)
        else:
            self.register_parameter('residual_linear', None)

        # Some fixed function
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_src.weight)
        nn.init.xavier_uniform_(self.linear_dst.weight)

        nn.init.xavier_uniform_(self.double_attn)
        if self.residual:
            if self.num_node_features != self.num_heads * self.output_dim:
                nn.init.xavier_uniform_(self.residual_linear.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]):

        # Input preprocessing
        x, edge_index = data
        edge_src_index, edge_dst_index = edge_index

        # Projection on the new space
        src_projected = self.linear_src(self.dropout(x)).view(-1, self.num_heads, self.output_dim)
        dst_projected = self.linear_dst(self.dropout(x)).view(-1, self.num_heads, self.output_dim)
        
        # Edge attention coefficients
        edge_attn = self.leakyReLU((src_projected.index_select(0, edge_src_index)
                                    + dst_projected.index_select(0, edge_dst_index)))
        edge_attn = (self.double_attn * edge_attn).sum(-1)
        exp_edge_attn = (edge_attn - edge_attn.max()).exp()

        # sum the edge scores to destination node
        num_nodes = x.shape[0]
        edge_node_score_sum = torch.zeros([num_nodes, self.num_heads],
                                          dtype=exp_edge_attn.dtype,
                                          device=exp_edge_attn.device)
        edge_dst_index_broadcast = edge_dst_index.unsqueeze(-1).expand_as(exp_edge_attn)
        edge_node_score_sum.scatter_add_(0, edge_dst_index_broadcast, exp_edge_attn)

        # normalized edge attention
        # edge_attn shape = [num_edges, num_heads, 1]
        exp_edge_attn = exp_edge_attn / (edge_node_score_sum.index_select(0, edge_dst_index) + 1e-16)
        exp_edge_attn = self.dropout(exp_edge_attn).unsqueeze(-1)

        # summation from one-hop atom
        edge_x_projected = src_projected.index_select(0, edge_src_index) * exp_edge_attn
        edge_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                  dtype=exp_edge_attn.dtype,
                                  device=exp_edge_attn.device)
        edge_dst_index_broadcast = (edge_dst_index.unsqueeze(-1)).unsqueeze(-1).expand_as(edge_x_projected)
        edge_output.scatter_add_(0, edge_dst_index_broadcast, edge_x_projected)

        output = edge_output
        # residual, concat, bias, activation
        if self.residual:
            output += self.residual_linear(x).view(num_nodes, -1, self.output_dim)
        if self.concat:
            output = output.view(-1, self.num_heads * self.output_dim)
        else:
            output = output.mean(dim=1)

        if self.bias is not None:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output, edge_index




