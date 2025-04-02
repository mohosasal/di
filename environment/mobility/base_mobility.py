from abc import ABC, abstractmethod
from typing import Dict, Tuple

class WdModel(ABC):
    @abstractmethod
    def move(self):
        pass