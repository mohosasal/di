from abc import ABC, abstractmethod
from typing import List
from ...graph.base_graph import GraphInterface

class GNNActor(ABC):
    @abstractmethod
    def forward(self, graph_manager: GraphInterface, tasks: List['Task']) -> 'torch.Tensor':
        pass