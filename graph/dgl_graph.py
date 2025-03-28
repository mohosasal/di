import torch
import dgl
import numpy as np
from .base_graph import GraphInterface
from typing import Dict, List, Tuple

class GraphManager(GraphInterface):
    def __init__(self, config):
        self.config = config
        self.graph = self._build_graph()
    
    def _build_graph(self) -> dgl.DGLGraph:
        g = dgl.DGLGraph()
        num_wds, num_aps, num_servers = (self.config.mec_params[k] for k in ['num_wds', 'num_aps', 'num_servers'])
        total_nodes = num_wds + num_aps + num_servers
        g.add_nodes(total_nodes)
        edges = [(wd, ap) for wd in range(num_wds) for ap in range(num_wds, num_wds + num_aps)] + \
                [(ap, server) for ap in range(num_wds, num_wds + num_aps) 
                 for server in range(num_wds + num_aps, total_nodes)]
        src, dst = zip(*edges)
        g.add_edges(src, dst)
        g.ndata['feat'] = torch.tensor(
            [[np.random.uniform(*self.config.mec_params['f_wd_range']), 0.0, 0] for _ in range(num_wds)] +
            [[self.config.mec_params['f_ap'], 0.0, 1] for _ in range(num_aps)] +
            [[self.config.mec_params['f_server'], 0.0, 2] for _ in range(num_servers)],
            dtype=torch.float32)
        g.edata['feat'] = torch.zeros(g.num_edges(), dtype=torch.float32)
        return g
    
    def get_node_features(self) -> List[List[float]]:
        return self.graph.ndata['feat'].tolist()
    
    def get_edge_features(self) -> List[float]:
        return self.graph.edata['feat'].tolist()
    
    def update_edge_rates(self, positions: Dict[int, Tuple[float, float]]):
        for edge_id, (src, dst) in enumerate(zip(*self.graph.edges())):
            if src in positions and dst in positions:
                dist = np.sqrt((positions[src][0] - positions[dst][0])**2 + 
                               (positions[src][1] - positions[dst][1])**2)
                H = dist**(-4)
                B = self.config.mec_params['bandwidth']
                P = self.config.mec_params['transmit_power']
                N = self.config.mec_params['noise']
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