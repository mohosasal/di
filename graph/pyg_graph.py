import torch
from torch_geometric.data import Data

import numpy as np

from di.environment.wireless_device import WirelessDevice


class PyGGraphManager:
    def __init__(self, wireless_device, config):
        self.config = config

        self.node_features = {}
        self.edge_features = {}

        self.node_indices = {}  # maps node_id -> index in PyG
        self.pyg_data = None

        self._build_graph()

    def _build_graph(self):
        vehicles = WirelessDevice.vehicles
        wd_ids = [vehicle.ID for vehicle in vehicles]

        # Map WD ids to PyG indices
        self.node_indices = {wd_id: idx for idx, wd_id in enumerate(wd_ids)}

        node_feats = []
        edge_index = []
        edge_attr = []

        # Build node features
        for wd_id in wd_ids:
            freq = np.random.uniform(*self.config.mec_params['f_wd_range'])
            feature = [freq, 0.0, 0]  # [frequency, reserved, type=0 for WD]
            self.node_features[wd_id] = feature
            node_feats.append(feature)

        # Connect WD to WD if within threshold
        for i in range(len(wd_ids)):
            for j in range(i + 1, len(wd_ids)):
                id1, id2 = wd_ids[i], wd_ids[j]
                pos1 = (vehicles[id1].x, vehicles[id1].y)
                pos2 = (vehicles[id2].x, vehicles[id2].y)

                dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                if dist <= self.config.wd_to_wd_threshold:
                    rate = 1.0 / (dist + 1e-6)

                    idx1 = self.node_indices[id1]
                    idx2 = self.node_indices[id2]

                    # Add edges both ways
                    edge_index.extend([[idx1, idx2], [idx2, idx1]])
                    edge_attr.extend([[rate], [rate]])

                    self.edge_features[(id1, id2)] = rate
                    self.edge_features[(id2, id1)] = rate

        # Convert everything to PyTorch tensors
        x = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Store as PyG data object
        self.pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def get_node_index(self, node_id):
        return self.node_indices.get(node_id)

    def get_node_feature(self, node_id):
        return self.node_features.get(node_id)

    def get_edge_feature(self, node1_id, node2_id):
        return self.edge_features.get((node1_id, node2_id))

    def update_wd_connections_and_rates(self):
        """
        Updates edge connections and rates between WDs only, based on a distance threshold.
        The graph is rebuilt in PyG format (not via NetworkX).
        """
        edge_index = []
        edge_attr = []

        wd_ids = list(self.wds.keys())
        node_idx_map = {wd_id: idx for idx, wd_id in enumerate(wd_ids)}
        positions = [self.wds[wd_id].pos for wd_id in wd_ids]

        for i, id1 in enumerate(wd_ids):
            for j in range(i + 1, len(wd_ids)):
                id2 = wd_ids[j]
                pos1, pos2 = positions[i], positions[j]
                distance = np.linalg.norm(pos1 - pos2)
                if distance <= self.config.wd_to_wd_threshold:
                    # Add edges in both directions
                    edge_index.append([node_idx_map[id1], node_idx_map[id2]])
                    edge_index.append([node_idx_map[id2], node_idx_map[id1]])

                    rate = self.calculate_edge_rate(self.wds[id1], self.wds[id2])
                    edge_attr.append([rate])
                    edge_attr.append([rate])  # symmetric rate for undirected edge

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        self.pyg_data.edge_index = edge_index
        self.pyg_data.edge_attr = edge_attr

