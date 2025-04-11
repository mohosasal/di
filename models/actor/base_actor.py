from abc import ABC, abstractmethod
from typing import List
from di.graph.base_graph import IGraphManager

class GNNActor(ABC):
    @abstractmethod
    def forward(self, graph_manager: IGraphManager, tasks: List['Task']) -> 'torch.Tensor':
        pass