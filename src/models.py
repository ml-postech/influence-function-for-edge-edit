import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptTensor,
)
from torch_geometric.utils import spmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm, GCNConv
from torch_geometric.nn.conv.cheb_conv import ChebConv
import torch.nn.functional as F
from src.gatconv import EdgeWeightedGATConv
import torch.nn as nn

class SGCConv(MessagePassing):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    
class SGC(torch.nn.Module):
    def __init__(self, in_dim, num_classes, num_layers=2):
        super().__init__()
        self.conv = SGCConv()
        self.num_layers = num_layers
        self.lin = torch.nn.Linear(in_features=in_dim, out_features=num_classes, bias=True)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight
        for _ in range(self.num_layers):
            x = self.conv(x, edge_index, edge_weight)
        x = self.lin(x)

        return x
    
class ChebNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, linear=False, bias=True):
        super().__init__()
        assert num_layers >= 2
        self.linear = linear

        self.convs = torch.nn.ModuleList()
        self.convs.append(ChebConv(in_dim, hidden_dim, K=2, bias=bias))
        for i in range(num_layers-2):
            self.convs.append(ChebConv(hidden_dim, hidden_dim, K=2, bias=bias))
        self.convs.append(ChebConv(hidden_dim, num_classes, K=2, bias=bias))

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight
        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index, edge_weight)
            if not self.linear:
                x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_weight)

        return x


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, linear=False, bias=True):
        super().__init__()
        assert num_layers >= 2
        self.linear = linear

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim, bias=bias))
        for i in range(num_layers-2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, bias=bias))
        self.convs.append(GCNConv(hidden_dim, num_classes, bias=bias))

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight
        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index, edge_weight)
            if not self.linear:
                x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_weight)

        return x
    
class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, heads=1, concat=True, linear=False, bias=True):
        super().__init__()
        assert num_layers >= 2
        self.linear = linear
        self.concat = concat
        self.heads = heads

        self.convs = nn.ModuleList()

        self.convs.append(EdgeWeightedGATConv(in_dim, hidden_dim, heads=heads, concat=concat, bias=bias))
        #self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=concat, bias=bias))

        for _ in range(num_layers - 2):
            in_dim_mid = hidden_dim * heads if concat else hidden_dim
            self.convs.append(EdgeWeightedGATConv(in_dim_mid, hidden_dim, heads=heads, concat=concat, bias=bias))
            #self.convs.append(GATConv(in_dim_mid, hidden_dim, heads=heads, concat=concat, bias=bias))

        in_dim_last = hidden_dim * heads if concat else hidden_dim
        self.convs.append(EdgeWeightedGATConv(in_dim_last, num_classes, heads=1, concat=False, bias=bias))
        #self.convs.append(GATConv(in_dim_last, num_classes, heads=1, concat=False, bias=bias))

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            if not self.linear:
                x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_weight)

        return x

class GNN(torch.nn.Module):
    def __init__(self, name, in_dim, hidden_dim, num_classes, num_layers=2, linear=False, bias=True, num_heads=1):
        super().__init__()
        
        if name == 'SGC':
            self.model = SGC(in_dim, num_classes, num_layers)
        elif name == 'GCN':
            self.model = GCN(in_dim, hidden_dim, num_classes, num_layers, linear, bias)
        elif name == 'GAT':
            self.model = GAT(in_dim, hidden_dim, num_classes, num_layers, num_heads, concat=True, linear=False, bias=False)
        elif name == 'ChebNet':
            self.model = ChebNet(in_dim, hidden_dim, num_classes, num_layers, linear=False, bias=False)

    def forward(self, graph):
        return self.model(graph)
    
