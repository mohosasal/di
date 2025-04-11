from typing import List
import torch
import torch.nn as nn
import numpy as np
from di.models.actor.base_actor import GNNActor
from di.graph.networkx_graph import NetworkXGraphManager


class NetworkXActor(GNNActor):
    def __init__(self, in_feats: int, hidden_feats: int):
        super().__init__()
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        
        # Define the neural network layers
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, graph_manager: NetworkXGraphManager, tasks: List['Task']) -> torch.Tensor:
        """
        Forward pass using NetworkX graph structure.
        
        This implementation uses a simplified approach that doesn't rely on graph convolutions
        but still considers the graph structure for feature aggregation.
        """
        # Get node features
        node_features = graph_manager.get_node_features()
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        
        # Get the graph structure
        graph = graph_manager.graph
        
        # Process each node
        num_nodes = len(node_features)
        hidden_states = torch.zeros(num_nodes, self.hidden_feats)
        
        # First layer: aggregate neighbor features and apply transformation
        for node in range(num_nodes):
            # Get neighbor features
            neighbors = list(graph.predecessors(node)) + list(graph.successors(node))
            if not neighbors:
                # If no neighbors, just use the node's own features
                neighbor_features = node_features_tensor[node].unsqueeze(0)
            else:
                # Average neighbor features
                neighbor_features = torch.mean(node_features_tensor[neighbors], dim=0)
            
            # Combine node features with aggregated neighbor features
            combined_features = torch.cat([node_features_tensor[node], neighbor_features])
            
            # Apply first layer transformation
            hidden_states[node] = torch.relu(self.fc1(combined_features))
        
        # Second layer: transform hidden states
        output = self.fc2(hidden_states)
        
        # Apply sigmoid to get probabilities
        probs = self.sigmoid(output[:len(tasks)])
        
        return probs.squeeze() 