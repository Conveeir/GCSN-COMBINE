import dgl
import torch
from torch import nn

import dgl.nn.pytorch as dglnn


class MyGCNLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(MyGCNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.drop_out = dropout
        self.conv = dglnn.GraphConv(hidden_size, hidden_size)
        self.w = torch.randn(hidden_size, hidden_size)

    def forward(self, graph, adj_matrix_ba, device):
        self.w = self.w.to(device)

        nodes_features = graph.ndata["h"]

        # =========================
        nodes_features = self.conv(graph, nodes_features)
        adj_ba_x = torch.mm(adj_matrix_ba, nodes_features)
        adj_ba_x_w = torch.mm(adj_ba_x, self.w)
        layer_out = torch.relu(adj_ba_x_w)
        # =========================

        graph.ndata['h'] = layer_out

        return graph
