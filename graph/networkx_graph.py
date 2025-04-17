import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
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

    def _shannon_capacity(self, dist: float, bandwidth: float = 1.0, power: float = 1.0, noise: float = 1.0) -> float:
        if dist == 0:
            return float('inf')  # Infinite capacity at zero distance
        snr = power / (noise * dist ** 2)
        return bandwidth * np.log2(1 + snr)

    def build_graph(self):
        # Initially build graph from scratch
        for dev_a in self.devices.values():
            for dev_b in self.devices.values():
                if dev_a.get_id() == dev_b.get_id():
                    continue
                id_a = dev_a.get_id()
                id_b = dev_b.get_id()
                pos_a = np.array(dev_a.get_position())
                pos_b = np.array(dev_b.get_position())
                dist = self._compute_distance(pos_a, pos_b)
                if dist <= self.threshold:
                    capacity = self._shannon_capacity(dist)
                    self.graph.add_edge(id_a, id_b, weight=capacity)

        self.visualize_graph()

    def update_edge_rates(self):
        current_edges = set(self.graph.edges)
        updated_edges = set()

        # Check all possible device pairs
        for dev_a in self.devices.values():
            for dev_b in self.devices.values():
                id_a = dev_a.get_id()
                id_b = dev_b.get_id()
                if id_a == id_b:
                    continue

                pos_a = np.array(dev_a.get_position())
                pos_b = np.array(dev_b.get_position())
                dist = self._compute_distance(pos_a, pos_b)

                if dist <= self.threshold:
                    capacity = self._shannon_capacity(dist)
                    self.graph.add_edge(id_a, id_b, weight=capacity)
                    updated_edges.add((min(id_a, id_b), max(id_a, id_b)))  # normalized order
                else:
                    # If edge exists and distance is too far, remove it
                    if self.graph.has_edge(id_a, id_b):
                        self.graph.remove_edge(id_a, id_b)

        # Optional: Clean up any leftover edges (if needed)
        # but the above loop already handles both add and remove

    def get_node_features(self) -> List[List[float]]:
        return [self.devices[node_id].feature.tolist() for node_id in self.graph.nodes]

    def get_edge_features(self) -> List[float]:
        return [self.graph[u][v]['weight'] for u, v in self.graph.edges]

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
        pos_src = np.array(self.devices[src].get_position())
        pos_dst = np.array(self.devices[dst].get_position())
        dist = self._compute_distance(pos_src, pos_dst)
        capacity = self._shannon_capacity(dist)
        self.graph.add_edge(src, dst, weight=capacity)

    def remove_edges(self, edge_id: int):
        edges = list(self.graph.edges)
        if 0 <= edge_id < len(edges):
            self.graph.remove_edge(*edges[edge_id])

    def visualize_graph(self):
        pos = {device_id: np.array(self.devices[device_id].get_position()) for device_id in self.graph.nodes}

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_size=300, node_color='skyblue')
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray')
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_color='black')

        # Draw edge weights (Shannon capacity)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Wireless Devices Graph (Shannon Capacity)")
        plt.axis("off")
        plt.show()
