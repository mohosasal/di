import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from models.actor.base_actor import GNNActor


class PyGActor(GNNActor):
    def __init__(self, in_feats, hidden_feats):
        super(PyGActor, self).__init__(in_feats, hidden_feats)
        
        # Define the GNN layers
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        
        # Define the output layer
        self.out = nn.Linear(hidden_feats, 1)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, graph):
        """
        Forward pass through the GNN.
        
        Args:
            graph: PyG Data object containing node features, edge indices, and edge attributes
            
        Returns:
            torch.Tensor: Node probabilities for task assignment
        """
        # Get node features and edge indices
        x = graph.x
        edge_index = graph.edge_index
        
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output layer
        x = self.out(x)
        
        # Apply sigmoid to get probabilities
        x = self.sigmoid(x)
        
        return x 