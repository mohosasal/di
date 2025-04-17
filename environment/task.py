from enum import Enum, auto

class TaskStatus(Enum):
    PENDING = auto()
    TRANSMITTING = auto()
    PROCESSING = auto()
    COMPUTATION_COMPLETED = auto()
    ACK_TRANSMITTING = auto()
    ACK_RECEIVED = auto()
    FULL_COMPLETED = auto()
    FORWARD_CORRUPTED = auto()

class Task:
    def __init__(self, task_id: int, wd_id: int, data_size: float, compute_demand: float, deadline: float, current_location :int):
        self.ID = task_id
        self.wd_id = wd_id
        self.data_size = data_size
        self.compute_demand = compute_demand
        self.deadline = deadline
        self.status = TaskStatus.PENDING
        
        self.remaining_times = {
            'transmission': 0,
            'processing': 0,
        }
        
        self.current_location = current_location
        self.destination = None
        self.path_for = None
        self.path_back = None
