import torch
import dgl
import numpy as np
from .base_graph import IGraphManager
from typing import Dict, List, Tuple

from ..environment.mobility.base_wd import IWirelessDevice


class GraphManager(IGraphManager):
    def __init__(self, config):
        self.config = config
        self.graph = self._build_graph()

    def _build_graph(self) -> dgl.DGLGraph:
        g = dgl.DGLGraph()
        num_wds = self.config.mec_params['num_wds']
        num_aps = self.config.mec_params['num_aps']
        num_servers = self.config.mec_params['num_servers']
        total_nodes = num_wds + num_aps + num_servers

        g.add_nodes(total_nodes)

        # Create edges: WD->AP and AP->Server
        edges = [(wd, ap) for wd in range(num_wds)
                 for ap in range(num_wds, num_wds + num_aps)] + \
                [(ap, server) for ap in range(num_wds, num_wds + num_aps)
                 for server in range(num_wds + num_aps, total_nodes)]

        src, dst = zip(*edges)
        g.add_edges(src, dst)

        # Initialize node features
        # [frequency, reserved, type] where type: 0=WD, 1=AP, 2=Server
        g.ndata['feat'] = torch.tensor(
            [[np.random.uniform(*self.config.mec_params['f_wd_range']), 0.0, 0]
             for _ in range(num_wds)] +
            [[self.config.mec_params['f_ap'], 0.0, 1]
             for _ in range(num_aps)] +
            [[self.config.mec_params['f_server'], 0.0, 2]
             for _ in range(num_servers)],
            dtype=torch.float32)

        # Initialize edge features with zeros
        g.edata['feat'] = torch.zeros(g.num_edges(), dtype=torch.float32)
        return g

    def get_node_features(self) -> List[List[float]]:
        return self.graph.ndata['feat'].tolist()

    def get_edge_features(self) -> List[float]:
        return self.graph.edata['feat'].tolist()

    def update_edge_rates(self, wd: IWirelessDevice):
        """
        Update edge rates based on WirelessDevice positions.
        Only updates edges where both nodes are wireless devices.
        """
        for edge_id, (src, dst) in enumerate(zip(*self.graph.edges())):
            # Find vehicles corresponding to source and destination nodes
            src_vehicle = next((v for v in wd.vehicles if v.VehicleID == src), None)
            dst_vehicle = next((v for v in wd.vehicles if v.VehicleID == dst), None)

            # Only update if both nodes are valid wireless devices
            if src_vehicle and dst_vehicle:
                # Calculate Euclidean distance between vehicles
                dist = np.sqrt((src_vehicle.x - dst_vehicle.x) ** 2 +
                               (src_vehicle.y - dst_vehicle.y) ** 2)

                # Prevent division by zero
                if dist == 0:
                    dist = 0.0001  # Small epsilon value

                # Calculate channel gain using distance-based path loss
                H = dist ** (-4)

                # Get parameters from config
                B = self.config.mec_params['bandwidth']
                P = self.config.mec_params['transmit_power']
                N = self.config.mec_params['noise']

                # Update edge feature with Shannon capacity
                self.graph.edata['feat'][edge_id] = B * np.log2(1 + P * H / N)

    def successors(self, node_id: int) -> List[int]:
        return self.graph.successors(node_id).tolist()

    def edge_ids(self, src: int, dst: int) -> int:
        return self.graph.edge_ids(src, dst).item()

    def has_edges_between(self, src: int, dst: int) -> bool:
        return self.graph.has_edges_between(src, dst).item()

    def add_edges(self, src: int, dst: int):
        self.graph.add_edges(src, dst)

    def remove_edges(self, edge_id: int):
        self.graph.remove_edges(edge_id)

