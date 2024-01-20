from .dataset import GraphDataset, GraphData, GraphFeaturizer
from typing import List, Union
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import degree
import dgl


class GraphormerDataset(GraphDataset):
    def __init__(
        self,
        raw_path: str,
        label_path: str = None,
        remove_hydrogen: bool = False,
        label_col: Union[str, List[str]] = None,
        additional_cols: List[str] = [],
        featurizer: GraphFeaturizer = None,
        processed_path: str = None,
        force_reload: bool = False,
        save_processed: bool = False,
        toy_length: int = -1,
    ):
        super().__init__(
            raw_path,
            label_path,
            remove_hydrogen,
            label_col,
            additional_cols,
            featurizer,
            processed_path,
            force_reload,
            save_processed,
            toy_length
        )
    
    @staticmethod
    def collate_fn(batch: List[GraphData]):
        labels = torch.concatenate([data.y for data in batch], dim=0)

        num_graphs = len(batch)
        num_nodes = [data.num_nodes for data in batch]
        max_num_nodes = max(num_nodes)

        # Graphormer adds a virual node to the graph, which is connected to
        # all other nodes and supposed to represent the graph embedding. So
        # here +1 is for the virtual node.
        attn_mask = torch.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
        node_feat = []
        in_degree, out_degree = [], []
        path_data = []

        # Since shortest_dist returns -1 for unreachable node pairs and padded
        # nodes are unreachable to others, distance relevant to padded nodes
        # use -1 padding as well.
        dist = -torch.ones((num_graphs, max_num_nodes, max_num_nodes), 
                            dtype=torch.long)

        for i in range(num_graphs):
            graph = batch[i]
            
            # Generate shortest path distance matrix.
            dgl_graph = dgl.graph((graph.edge_index[0], graph.edge_index[1]))
            graph.spd, graph.path = dgl.shortest_dist(dgl_graph, 
                                                      root=None, 
                                                      return_paths=True)
            # A binary mask where invalid positions are indicated by True.
            attn_mask[i, :, num_nodes[i] + 1 :] = 1

            # +1 to distinguish padded non-existing nodes from real nodes
            node_feat.append(graph.x + 1)

            in_degree_ = degree(index=graph.edge_index[1], 
                                num_nodes=graph.num_nodes).long()
            in_degree.append(torch.clamp(in_degree_ + 1, min=0, max=512))
            out_degree_ = degree(index=graph.edge_index[0], 
                                 num_nodes=graph.num_nodes).long()
            out_degree.append(torch.clamp(out_degree_ + 1, min=0, max=512))

            # Path padding to make all paths to the same length "max_len".
            path = graph.path
            path_len = path.size(dim=2)
            # shape of shortest_path: [n, n, max_len]
            max_len = 5
            if path_len >= max_len:
                shortest_path = path[:, :, :max_len]
            else:
                p1d = (0, max_len - path_len)
                # Use the same -1 padding as shortest_dist for
                # invalid edge IDs.
                shortest_path = F.pad(path, p1d, "constant", -1)
            pad_num_nodes = max_num_nodes - num_nodes[i]
            p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
            shortest_path = F.pad(shortest_path, p3d, "constant", -1)
            # +1 to distinguish padded non-existing edges from real edges
            edata = graph.edge_attr + 1
            # shortest_dist pads non-existing edges (at the end of shortest
            # paths) with edge IDs -1, and th.zeros(1, edata.shape[1]) stands
            # for all padded edge features.
            edata = torch.cat(
                (edata, torch.zeros(1, edata.shape[1]).to(edata.device)), dim=0
            )
            path_data.append(edata[shortest_path])

            dist[i, : num_nodes[i], : num_nodes[i]] = graph.spd

        # node feat padding
        node_feat = pad_sequence(node_feat, batch_first=True)

        # degree padding
        in_degree = pad_sequence(in_degree, batch_first=True)
        out_degree = pad_sequence(out_degree, batch_first=True)
        
        batch_data = GraphData(
            y=labels.reshape(num_graphs, -1),
            attn_mask=attn_mask,
            x=node_feat,
            in_degree=in_degree,
            out_degree=out_degree,
            path_data=torch.stack(path_data),
            dist=dist,
        )
        
        return batch_data