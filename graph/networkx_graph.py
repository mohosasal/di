import networkx as nx
import numpy as np
from typing import List, Dict, Tuple

from di.graph.base_graph import IGraphManager
from di.environment.base_object import IWirelessDevice


class NetworkXManager(IGraphManager):
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.devices: Dict[int, IWirelessDevice] = {}
        self.graph = nx.Graph()

    def add_device(self, device: IWirelessDevice):
        device_id = device.get_id()
        self.devices[device_id] = device
        self.graph.add_node(device_id)

    def _compute_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def build_graph(self):
        """
        Builds the graph using NetworkX based on a global list of wireless devices
        available in WirelessDevice.vehicles.
        """
        # Clear the graph and device mapping
        self.graph.clear()
        self.devices.clear()

        # Import the class if not already (you can also do this at the top of the file)
        from di.environment.wireless_device import WirelessDevice

        # Register all vehicles into devices and graph
        for device in WirelessDevice.vehicles:
            device_id = device.get_id()
            self.devices[device_id] = device
            self.graph.add_node(device_id)

        # Build edges based on distance threshold
        for dev_a in WirelessDevice.vehicles:
            for dev_b in WirelessDevice.vehicles:
                if dev_a.get_id() == dev_b.get_id():
                    continue
                dist = self._compute_distance(dev_a.get_position(), dev_b.get_position())
                if dist <= self.threshold:
                    self.graph.add_edge(dev_a.get_id(), dev_b.get_id(), weight=1.0)  # Placeholder edge feature


    # Implementing IGraphManager methods:
    def get_node_features(self) -> List[List[float]]:
        return [self.devices[node_id].feature.tolist() for node_id in self.graph.nodes]

    def get_edge_features(self) -> List[float]:
        return [self.graph[u][v]['weight'] for u, v in self.graph.edges]

    def update_edge_rates(self):
        self.build_graph()

    def successors(self, node_id: int) -> List[int]:
        return list(self.graph.neighbors(node_id))

    def edge_ids(self, src: int, dst: int) -> int:
        try:
            edges = list(self.graph.edges)
            return edges.index((src, dst)) if (src, dst) in edges else edges.index((dst, src))
        except ValueError:
            return -1

    def has_edges_between(self, src: int, dst: int) -> bool:
        return self.graph.has_edge(src, dst)

    def add_edges(self, src: int, dst: int):
        self.graph.add_edge(src, dst, weight=1.0)

    def remove_edges(self, edge_id: int):
        edges = list(self.graph.edges)
        if 0 <= edge_id < len(edges):
            self.graph.remove_edge(*edges[edge_id])
