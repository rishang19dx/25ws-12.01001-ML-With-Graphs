import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool


class GNNLayerF(nn.Module):
    """GNN Layer f: Sum aggregation"""
    def __init__(self, in_dim, out_dim, activation='relu'):
        super(GNNLayerF, self).__init__()
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x, edge_index):
        self_features = self.W1(x)
        row, col = edge_index
        neighbor_features = self.W2(x)
        aggregated = torch.zeros(x.size(0), self.W2.out_features, device=x.device, dtype=x.dtype)
        aggregated.index_add_(0, row, neighbor_features[col])
        out = self.activation(self_features + aggregated)
        return out


class GNNLayerG(nn.Module):
    """GNN Layer g: Mean aggregation"""
    def __init__(self, in_dim, out_dim, activation='relu'):
        super(GNNLayerG, self).__init__()
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Identity()

    def forward(self, x, edge_index):
        self_features = self.W1(x)
        row, col = edge_index
        neighbor_features = self.W2(x)

        degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        degree.index_add_(0, row, torch.ones(row.size(0), device=x.device))
        degree = degree.clamp(min=1)

        aggregated = torch.zeros(x.size(0), self.W2.out_features, device=x.device, dtype=x.dtype)
        aggregated.index_add_(0, row, neighbor_features[col])
        aggregated = aggregated / degree.unsqueeze(1)
        out = self.activation(self_features + aggregated)
        return out


class GNN(nn.Module):
    """GNN model with configurable layers, pooling, and activation"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3,
                 layer_type='f', activation='relu', pooling='mean', dropout=0.0):
        super(GNN, self).__init__()
        LayerClass = GNNLayerF if layer_type == 'f' else GNNLayerG

        self.layers = nn.ModuleList()
        self.layers.append(LayerClass(input_dim, hidden_dim, activation))
        for _ in range(num_layers - 1):
            self.layers.append(LayerClass(hidden_dim, hidden_dim, activation))

        self.dropout = nn.Dropout(dropout)
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x