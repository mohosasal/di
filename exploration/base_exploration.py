from abc import ABC, abstractmethod
from typing import List
from ..graph.base_graph import GraphInterface

class ExplorationStrategy(ABC):
    @abstractmethod
    def explore(self, predicted_actions: 'torch.Tensor', graph_manager: GraphInterface, 
                max_hops: int) -> List[List[int]]:
        pass