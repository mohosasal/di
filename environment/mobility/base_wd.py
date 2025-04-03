from typing import List, Dict, Protocol

from di.environment.task import Task


class IWirelessDevice(Protocol):
    @staticmethod
    def move_all() -> None:
        pass

    @staticmethod
    def populate() -> None:
        pass

    def process_local_task(self, task: Task, queue_delay: float) -> float:
        pass

    @property
    def vehicles(self) -> List:
        pass
    @staticmethod
    def get_vehicle_by_id(wd_id):
        pass