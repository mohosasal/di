import torch
import dgl
import numpy as np
from .base_graph import IGraphManager
from typing import List, Dict
from di.environment.base_object import IWirelessDevice


class GraphManager(IGraphManager):
    threshold = 1
    def __init__(self, config, wds: List[IWirelessDevice], aps: List[int], servers: List[int]):

        self.config = config
        self.wds = {wd.ID: wd for wd in wds}
        self.aps = {ap.ID: ap for ap in aps}
        self.servers = {server.ID: server for server in servers}

        # Create node mapping
        self.node_map = {**{wd.ID: wd.ID for wd in wds},
                         **{ap: ap for ap in aps},
                         **{server: server for server in servers}}

        # Create the graph dynamically
        self.graph = self._build_graph()

    def _build_graph(self) -> dgl.DGLGraph:
        g = dgl.DGLGraph()

        # Add nodes using actual IDs
        all_nodes = list(self.node_map.keys())
        g.add_nodes(len(all_nodes))

        # Define edges dynamically
        edges = []

        # WD → AP connections
        for wd_id in self.wds.keys():
            for ap in self.aps:
                edges.append((wd_id, ap))

        # AP → Server connections
        for ap in self.aps:
            for server in self.servers:
                edges.append((ap, server))

        # **C2C: Direct WD → Server connections**
        for wd_id, wd in self.wds.items():
            if wd.C2C:  # If C2C is enabled, allow direct transmission
                for server in self.servers:
                    edges.append((wd_id, server))

        # Add edges to the graph
        if edges:
            src, dst = zip(*edges)
            g.add_edges(src, dst)

        # Initialize node features: [frequency, reserved, type] (0=WD, 1=AP, 2=Server)
        g.ndata['feat'] = torch.tensor(
            [[np.random.uniform(*self.config.mec_params['f_wd_range']), 0.0, 0] for _ in self.wds] +
            [[self.config.mec_params['f_ap'], 0.0, 1] for _ in self.aps] +
            [[self.config.mec_params['f_server'], 0.0, 2] for _ in self.servers],
            dtype=torch.float32
        )

        # Initialize edge features with zeros
        g.edata['feat'] = torch.zeros(g.num_edges(), dtype=torch.float32)

        return g

    def get_node_features(self) -> List[List[float]]:
        return self.graph.ndata['feat'].tolist()

    def get_edge_features(self) -> List[float]:
        return self.graph.edata['feat'].tolist()

    def update_edge_rates(self, wd: IWirelessDevice):
        for edge_id, (src, dst) in enumerate(zip(*self.graph.edges())):
            src_vehicle = self.wds.get(src, None)
            dst_vehicle = self.wds.get(dst, None)

            if src_vehicle and dst_vehicle:
                dist = np.sqrt((src_vehicle.x - dst_vehicle.x) ** 2 +
                               (src_vehicle.y - dst_vehicle.y) ** 2)
                dist = max(dist, 0.0001)  # Avoid division by zero

                # Path loss model
                H = dist ** (-4)
                B = self.config.mec_params['bandwidth']
                P = self.config.mec_params['transmit_power']
                N = self.config.mec_params['noise']

                # Shannon capacity update
                rate = B * np.log2(1 + P * H / N)
                self.graph.edata['feat'][edge_id] = rate

                if rate < GraphManager.threshold:
                    self.graph.remove_edges(edge_id)

    def successors(self, node_id: int) -> List[int]:
        return self.graph.successors(node_id).tolist()

    def edge_ids(self, src: int, dst: int) -> int:
        return self.graph.edge_ids(src, dst).item()

    def has_edges_between(self, src: int, dst: int) -> bool:
        return self.graph.has_edges_between(src, dst).item()

    def add_edges(self, src: int, dst: int):
        """ Add an edge dynamically """
        self.graph.add_edges(src, dst)

    def remove_edges(self, edge_id: int):
        """ Remove an edge by its ID """
        self.graph.remove_edges(edge_id)
