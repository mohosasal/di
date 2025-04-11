from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from di.environment.base_object import IWirelessDevice


class IGraphManager(ABC):
    @abstractmethod
    def get_node_features(self) -> List[List[float]]:
        pass
    
    @abstractmethod
    def get_edge_features(self) -> List[float]:
        pass
    
    @abstractmethod
    def update_edge_rates(self):
        pass
    
    @abstractmethod
    def successors(self, node_id: int) -> List[int]:
        pass
    
    @abstractmethod
    def edge_ids(self, src: int, dst: int) -> int:
        pass
    
    @abstractmethod
    def has_edges_between(self, src: int, dst: int) -> bool:
        pass
    
    @abstractmethod
    def add_edges(self, src: int, dst: int):
        pass
    
    @abstractmethod
    def remove_edges(self, edge_id: int):
        pass