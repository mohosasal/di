from abc import ABC, abstractmethod
from typing import Dict, Tuple

class MobilityModel(ABC):
    @abstractmethod
    def move(self, positions: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
        pass