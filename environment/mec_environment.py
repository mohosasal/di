from typing import List, Dict, Set
from collections import defaultdict
from pwnlib.dynelf import sizeof
from di.environment.base_object import IWirelessDevice
from di.environment.task import Task, TaskStatus
from di.graph.base_graph import IGraphManager


class MECEnvironment:
    def __init__(self, config, graph_manager: IGraphManager, wireless_device: IWirelessDevice):
        self.config = config
        self.graph_manager = graph_manager
        self.tasks: List[Task] = []
        self.queues: Dict[int, List[Task]] = defaultdict(list)
        self.graph_manager.update_edge_rates()
        self.wireless_device = wireless_device

    def step(self):

        min_time = self._find_min_remaining_time()

        self.move_vehicles(min_time)
        self.update_topology()
        self.process_tasks(min_time)

        return min_time

    def _find_min_remaining_time(self) -> float:

        if not self.tasks:
            return float('inf')

        min_time = float('inf')
        for task in self.tasks:
            active_time = self._get_active_time_component(task)
            if 0 < active_time < min_time:
                min_time = active_time

        return min_time

    def _get_active_time_component(self, task: Task) -> float:

        if task.status == TaskStatus.TRANSMITTING:
            return task.remaining_times['transmission']
        elif task.status == TaskStatus.PROCESSING:
            return task.remaining_times['processing']
        elif task.status == TaskStatus.ACK_TRANSMITTING:
            return task.remaining_times['ack_transmission']
        else:
            return float('inf')

    def move_vehicles(self, time):
        self.wireless_device.move_all(time)

    def update_topology(self):
        self.graph_manager.update_edge_rates()

    def process_tasks(self, time_step: float = 1.0):

        for task in self.tasks:
            if task.status == TaskStatus.TRANSMITTING:
                self.forward_task(task, time_step)
            elif task.status == TaskStatus.PROCESSING:
                self.compute_task(task, time_step)
            elif task.status == TaskStatus.COMPUTATION_COMPLETED:
                self.generate_ack(task)
            elif task.status == TaskStatus.ACK_TRANSMITTING:
                self.forward_task(task, time_step)
            elif task.status == TaskStatus.ACK_RECEIVED:
                pass
            elif task.status == TaskStatus.FORWARD_CORRUPTED:
                pass

    def forward_task(self, task: Task, time_step: float = 1.0):
        task.remaining_times['transmission'] -= time_step

        if task.remaining_times['transmission'] <= 0:
            next_node = self.get_next_hop(task)

            if next_node == task.destination:
                # --- Arrival at destination ---
                if task.status == TaskStatus.ACK_TRANSMITTING:
                    # ACK arrived at original sender
                    task.current_location = task.destination
                    task.status = TaskStatus.ACK_RECEIVED

                    # === DAG Dependency Logic: Release children tasks if eligible ===
                    for child in getattr(task, 'dependents', []):
                        # Eligible if ALL parents are finished (ACK_RECEIVED or FULL_COMPLETED)
                        if all(parent.status in TaskStatus.ACK_RECEIVED for parent in
                               getattr(child, 'dependencies', [])):
                            if child.status == TaskStatus.PENDING:
                                child.status = TaskStatus.TRANSMITTING
                                child.remaining_times['transmission'] = self.estimate_transmission_time(child,
                                                                                                        self.get_next_hop(
                                                                                                            child))
                    # =================================================================

                else:
                    task.status = TaskStatus.PROCESSING
                    task.current_location = task.destination
                    available_resources = self.get_available_resources(task.current_location)
                    task.remaining_times['processing'] = task.compute_demand / available_resources

            else:
                if self.graph_manager.has_edges_between(task.current_location, next_node):
                    task.current_location = next_node
                    new_transmission_time = self.estimate_transmission_time(task, next_node)
                    task.remaining_times['transmission'] = new_transmission_time
                else:
                    task.status = TaskStatus.FORWARD_CORRUPTED

    def compute_task(self, task: Task, time_step: float = 1.0):

        # todo should compute dag task

        available_resources = self.get_available_resources(task.current_location)
        task.remaining_times['processing'] -= available_resources * time_step
        task.remaining_times['total'] -= available_resources * time_step

        if task.remaining_times['processing'] <= 0:
            task.status = TaskStatus.COMPUTATION_COMPLETED
            self.generate_ack(task)

    def generate_ack(self, task: Task):
        task.data_size = self.config.ack_size
        task.compute_demand = 0
        task.status = TaskStatus.ACK_TRANSMITTING
        task.destination = task.path_back[sizeof(task.path_back) - 1]

        ack_transmission_time = self.estimate_transmission_time(task, task.wd_id)
        task.remaining_times['transmission'] = ack_transmission_time

    def get_available_resources(self, node_id: int) -> float:
        return self.graph_manager.get_node_features()[node_id][0]

    def get_next_hop(self, task: Task) -> int:

        if task.status != TaskStatus.ACK_TRANSMITTING:
            path = task.path_for
        else:
            path = task.path_back
        if task.current_location in path:
            index = path.index(task.current_location)
            if index + 1 < len(path):
                return path[index + 1]
        return task.destination

    def estimate_transmission_time(self, task: Task, next_node: int) -> float:

        edge_id = self.graph_manager.edge_ids(task.current_location, next_node)

        if  edge_id== -1 :
            return float("inf")

        features = self.graph_manager.get_edge_features()
        # pick the best (max) capacity among them
        max_capacity = max(features[eid] for eid in [edge_id])

        # 4) if capacity is non‐positive, also treat as unreachable
        if max_capacity <= 0:
            return float("inf")

        # 5) normal case: data_size / capacity
        return task.data_size / max_capacity

    def reset(self):
        self.tasks.clear()
        self.queues.clear()
        self.wireless_device.populate()
        self.graph_manager.update_edge_rates(self.wireless_device)

    def load_sample_tasks(self):

        wd_id = self.wireless_device.get_all()[0].id

        task = Task(
            task_id=0,
            wd_id=wd_id,
            data_size=5.0,  # 5 MB
            compute_demand=2000,  # 2000 CPU cycles
            deadline=20.0,  # 20 seconds
            current_location=wd_id
        )

        # Set destination and path
        task.destination = self.wireless_device.get_all()[2].id
        task.path_for = [wd_id, self.wireless_device.get_all()[1].id,
                         self.wireless_device.get_all()[2].id]  # Simple direct path
        task.path_back = [self.wireless_device.get_all()[2].id, self.wireless_device.get_all()[1].id,
                          wd_id]  # Return path

        task.status = TaskStatus.TRANSMITTING
        task.remaining_times['transmission'] = 10

        self.tasks.append(task)
        return self.tasks

        ###### important non-implemented issues :

        # transmitting to ap
        # server
        # ...
        # fp
        # retry mechanism if the path corrupted !
        # always check where to stop step
        # temporal GNN
        # Graph Sage


    # todo analyze the parameters of the simulation in timestamps

    # todo add quality of service outputs
    # -- child and parent tasks release and offload and ... timestamps
