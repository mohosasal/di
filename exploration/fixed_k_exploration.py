import numpy as np
from .base_exploration import ExplorationStrategy
from ..graph.base_graph import GraphInterface

class FixedKExploration(ExplorationStrategy):
    def __init__(self, k: int):
        self.k = k
    
    def explore(self, predicted_actions: 'torch.Tensor', graph_manager: GraphInterface, 
                max_hops: int) -> List[List[int]]:
        num_wds = len(predicted_actions)
        actions = []
        for _ in range(self.k):
            action_list = []
            for wd in range(num_wds):
                if predicted_actions[wd] < 0.5 or np.random.random() < 0.3:
                    action_list.append([wd])
                else:
                    path = [wd]
                    curr = wd
                    for _ in range(max_hops - 1):
                        neighbors = graph_manager.successors(curr)
                        if not neighbors:
                            break
                        curr = np.random.choice(neighbors)
                        path.append(curr)
                    action_list.append(path)
            actions.append(action_list)
        return actions