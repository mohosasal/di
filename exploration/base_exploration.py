from abc import ABC, abstractmethod
from typing import List
from di.graph.base_graph import IGraphManager

class ExplorationStrategy(ABC):
    @abstractmethod
    def explore(self, predicted_actions: 'torch.Tensor', graph_manager: IGraphManager,
                max_hops: int) -> List[List[int]]:
        pass


    #