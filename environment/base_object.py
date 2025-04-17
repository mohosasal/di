# interface.py
from typing import List, Protocol, Tuple
from di.environment.task import Task


class IWirelessDevice(Protocol):

    def __init__(self):
        self.feature = list()

    @staticmethod
    def load_data(file_path) -> None:
        pass

    @staticmethod
    def move_all(time) -> None:
        pass

    def move(self) -> None:
        pass

    @staticmethod
    def get_wd_by_id(id) -> "IWirelessDevice":
        pass

    def get_position(self) -> Tuple[float, float]:
        pass

    def get_id(self) -> str:
        pass

    @staticmethod
    def get_all() -> List["IWirelessDevice"]:
        pass
