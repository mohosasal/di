from typing import List, Dict, Tuple
from collections import defaultdict
from ..graph.base_graph import GraphInterface
from .mobility.base_mobility import MobilityModel
from task import *

class MECEnvironment:
    def __init__(self, config, graph_manager: GraphInterface, mobility_model: MobilityModel):
        self.config = config
        self.graph_manager = graph_manager
        self.mobility_model = mobility_model
        self.tasks: List[Task] = []
        self.queues: Dict[int, List[Task]] = defaultdict(list)
        self.positions: Dict[int, Tuple[float, float]] = self._init_positions()
        self.graph_manager.update_edge_rates(self.positions)
    
    def _init_positions(self) -> Dict[int, Tuple[float, float]]:
        num_wds = self.config.mec_params['num_wds']
        num_aps = self.config.mec_params['num_aps']
        positions = {i: (np.random.uniform(0, 100), np.random.uniform(0, 100)) for i in range(num_wds)}
        for i in range(num_wds, num_wds + num_aps):
            positions[i] = (50 + 20 * (i - num_wds), 50)
        return positions
    
    def generate_tasks(self):
        import numpy as np  # Only numpy, no torch/dgl
        self.tasks = [Task(np.random.uniform(*self.config.mec_params['d_i_range']),
                           np.random.uniform(*self.config.mec_params['k_i_range']),
                           np.random.choice([1, 2, 3]), 1.0, i) 
                      for i in range(self.config.mec_params['num_wds'])]
    
    def step(self, actions: List[List[int]]) -> float:
        import numpy as np
        self.queues.clear()
        total_latency = 0.0
        node_features = self.graph_manager.get_node_features()
        edge_features = self.graph_manager.get_edge_features()
        
        for task, path in zip(self.tasks, actions):
            final_node = path[-1] if len(path) > 1 else task.wd_id
            self.queues[final_node].append(task)
        
        for node_id, queue in self.queues.items():
            for task in queue:
                path = next(p for p, t in zip(actions, self.tasks) if t.wd_id == task.wd_id)
                if len(path) == 1:  # Local
                    T_l = task.k_i / node_features[node_id][0] + node_features[node_id][1]
                    total_latency += task.gamma_i * T_l
                else:  # Multi-hop
                    T_trans = sum(task.d_i / edge_features[self.graph_manager.edge_ids(path[i], path[i + 1])]
                                  for i in range(len(path) - 1))
                    T_edge = task.k_i / node_features[node_id][0]
                    total_latency += task.gamma_i * (T_trans + T_edge)
            node_features[node_id][1] = len(queue) * 0.01
        
        self.positions = self.mobility_model.move(self.positions)
        self.graph_manager.update_edge_rates(self.positions)
        
        if np.random.random() < 0.05:
            wd = np.random.randint(0, self.config.mec_params['num_wds'])
            ap = np.random.randint(self.config.mec_params['num_wds'], 
                                  self.config.mec_params['num_wds'] + self.config.mec_params['num_aps'])
            if self.graph_manager.has_edges_between(wd, ap):
                self.graph_manager.remove_edges(self.graph_manager.edge_ids(wd, ap))
            else:
                self.graph_manager.add_edges(wd, ap)
                self.graph_manager.update_edge_rates(self.positions)
        
        return total_latency