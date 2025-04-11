from typing import List, Dict
from di.graph.base_graph import IGraphManager

class Critic:
    def evaluate(self, graph_manager: IGraphManager, tasks: List['Task'],
                 actions: List[List[int]], queues: Dict[int, List['Task']]) -> float:
        total_latency = 0.0
        node_features = graph_manager.get_node_features()
        edge_features = graph_manager.get_edge_features()
        
        for task, path in zip(tasks, actions):
            final_node = path[-1] if len(path) > 1 else task.wd_id
            queue = queues.get(final_node, [])
            if len(path) == 1:
                T_l = task.k_i / node_features[final_node][0] + node_features[final_node][1]
                total_latency += task.gamma_i * T_l
            else:
                T_trans = sum(task.d_i / edge_features[graph_manager.edge_ids(path[i], path[i + 1])]
                              for i in range(len(path) - 1))
                T_edge = task.k_i / node_features[final_node][0]
                total_latency += task.gamma_i * (T_trans + T_edge)
        return total_latency