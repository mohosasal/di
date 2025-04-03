from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"         # Task is created but not yet started
    TRANSMITTING = "transmitting" # Task is being transmitted to an edge node
    IN_PROGRESS = "in_progress"  # Task is being processed at an edge node
    COMPLETED = "completed"      # Task processing is completed
    ACK_PENDING = "ack_pending"  # Waiting for acknowledgment transmission
    ACK_RECEIVED = "ack_received" # Acknowledgment received, fully done


class Task:
    def __init__(self, wd_id: int, data_size: float, computation_load: float, gamma_i: float, is_ack: bool = False):
        """
        :param wd_id: ID of the originating wireless device
        :param data_size: Amount of data to be transmitted (MB)
        :param computation_load: Computational load required (CPU cycles)
        :param gamma_i: Weight factor for latency importance
        :param is_ack: Indicates if this is an acknowledgment task
        """
        self.wd_id = wd_id
        self.data_size = data_size
        self.computation_load = computation_load
        self.gamma_i = gamma_i
        self.status = TaskStatus.PENDING
        self.remaining_transmission = data_size  # Tracks remaining data to be transmitted
        self.is_ack = is_ack  # Acknowledgment tasks will be marked
        self.current_node: Optional[int] = None  # The current processing node
        self.source_node: Optional[int] = None  # The original sender

    def compute(self, available_resources: float) -> bool:
        """
        Process the task based on the available computational resources at the current node.
        :param available_resources: CPU cycles per second available at the node.
        :return: True if the task is completed in this step, False otherwise.
        """
        if self.status != TaskStatus.PROCESSING:
            return False  # Can't process if not in the correct state.

        self.remaining_computation -= available_resources  # Reduce remaining work
        if self.remaining_computation <= 0:
            self.status = TaskStatus.COMPLETED
            self.remaining_computation = 0
            return True  # Task finished this step
        return False  # Task still needs more steps to finish

    def start_transmission(self, destination_node: int):
        """
        Mark the task as being transmitted to a new destination.
        """
        self.status = TaskStatus.TRANSMITTING
        self.offloaded_to = destination_node  # Mark where it's going

    def acknowledge(self):
        """
        Mark the task as acknowledged (final step when the original device receives the ack).
        """
        self.status = TaskStatus.ACKNOWLEDGED
