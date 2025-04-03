from typing import List, Dict
from collections import defaultdict

from mobility.base_wd import IWirelessDevice
from task import Task, TaskStatus
from ..graph.base_graph import IGraphManager


class MECEnvironment:
    def __init__(self, config, graph_manager: IGraphManager, wd: IWirelessDevice):
        self.config = config
        self.graph_manager = graph_manager
        self.wireless_device = wd
        self.tasks: List[Task] = []
        self.queues: Dict[int, List[Task]] = defaultdict(list)
        self.graph_manager.update_edge_rates(self.wireless_device)

    def run_simulation(self, max_steps: int):
        for _ in range(max_steps):
            if self.all_tasks_completed():
                break  # Stop if no more tasks are left
            self.step()  # Move the system forward by one step

    def step(self, actions: List[List[int]]) -> float:
        self.wireless_device.move_all()
        self.graph_manager.update_edge_rates(self.wireless_device)

        self._process_tasks(actions)
        self._handle_transmitting_tasks()
        self._generate_ack_tasks()

        return self._calculate_total_latency()

    def _process_tasks(self, actions: List[List[int]]):
        for task, path in zip(self.tasks, actions):
            if task.status == TaskStatus.PENDING:
                if len(path) == 1:
                    task.status = TaskStatus.IN_PROGRESS
                    self._process_locally(task, path[0])
                else:
                    task.status = TaskStatus.TRANSMITTING
                    task.remaining_transmission_steps = self._calculate_transmission_steps(task, path)

    def _handle_transmitting_tasks(self):
        for task in self.tasks:
            if task.status == TaskStatus.TRANSMITTING:
                task.remaining_transmission_steps -= 1
                if task.remaining_transmission_steps <= 0:
                    task.status = TaskStatus.IN_PROGRESS
                    self._process_remotely(task)

    def _generate_ack_tasks(self):
        new_tasks = []
        for task in self.tasks:
            if task.status == TaskStatus.COMPLETED:
                ack_task = Task(task.wd_id, task.offload_source, task.size)
                ack_task.status = TaskStatus.ACK_PENDING
                new_tasks.append(ack_task)
                task.status = TaskStatus.ACK_RECEIVED
        self.tasks.extend(new_tasks)

    def _process_locally(self, task: Task, node_id: int):
        node_features = self.graph_manager.get_node_features()
        processing_time = task.compute(node_features[node_id][0])
        task.status = TaskStatus.COMPLETED if processing_time <= 1 else TaskStatus.IN_PROGRESS

    def _process_remotely(self, task: Task):
        node_features = self.graph_manager.get_node_features()
        processing_time = task.compute(node_features[task.final_destination][0])
        task.status = TaskStatus.COMPLETED if processing_time <= 1 else TaskStatus.IN_PROGRESS

    def _calculate_transmission_steps(self, task: Task, path: List[int]) -> int:
        edge_features = self.graph_manager.get_edge_features()
        total_time = sum(task.d_i / edge_features[self.graph_manager.edge_ids(path[i], path[i + 1])]
                         for i in range(len(path) - 1))
        return max(1, round(total_time))

    def _calculate_total_latency(self) -> float:
        return sum(
            task.latency for task in self.tasks if task.status in [TaskStatus.COMPLETED, TaskStatus.ACK_RECEIVED])

    def reset(self):
        self.tasks.clear()
        self.queues.clear()
        self.wireless_device.populate()
        self.graph_manager.update_edge_rates(self.wireless_device)
