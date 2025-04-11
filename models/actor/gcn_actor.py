from typing import List

import torch
import torch.nn as nn
import dgl.nn as dglnn
from .base_actor import GNNActor
from ...graph.dgl_graph import GraphManager

class GCNActor(GNNActor):
    def __init__(self, in_feats: int, hidden_feats: int):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, graph_manager: GraphManager, tasks: List['Task']) -> torch.Tensor:
        graph = graph_manager.graph
        h = graph.ndata['feat']
        h = torch.relu(self.conv1(graph, h))
        h = self.conv2(graph, h)
        probs = self.sigmoid(h[:len(tasks)])
        return probs.squeeze()