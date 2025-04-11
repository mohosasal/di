from typing import List, Dict
from collections import defaultdict
from di.environment.base_object import IWirelessDevice
from di.environment.task import Task, TaskStatus
from di.environment.wireless_device import WirelessDevice
from di.graph.base_graph import IGraphManager
import random


class MECEnvironment:
    def __init__(self, config, graph_manager: IGraphManager):
        self.config = config
        self.graph_manager = graph_manager
        self.tasks: List[Task] = []
        self.queues: Dict[int, List[Task]] = defaultdict(list)
        self.graph_manager.update_edge_rates()

    def step(self):

        #todo make it works for 1 seceond too
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
        WirelessDevice.move_all(time)

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

        # Check if transmission is complete
        if task.remaining_times['transmission'] <= 0:

            next_node = self.get_next_hop(task)

            if next_node == task.destination:
                task.status = TaskStatus.PROCESSING
                task.current_location = task.destination
                # caclulate remaining time
                available_resources = self.get_available_resources(task.current_location)
                # Update the remaining processing time
                task.remaining_times['processing'] = task.compute_demand /  available_resources

            else:
                if self.graph_manager.has_edges_between(task.current_location, next_node):
                    task.current_location = next_node
                    # Set new transmission time
                    new_transmission_time = self.estimate_transmission_time(task, next_node)
                    task.remaining_times['transmitting']=new_transmission_time
                else:
                    task.status = TaskStatus.FORWARD_CORRUPTED

    def compute_task(self, task: Task, time_step: float = 1.0):

        available_resources = self.get_available_resources(task.current_location)
        # Update the remaining processing time
        task.remaining_times['processing'] -= available_resources * time_step
        task.remaining_times['total'] -= available_resources * time_step
        
        # Check if processing is complete
        if task.remaining_times['processing'] <= 0:
            task.status = TaskStatus.COMPUTATION_COMPLETED
            self.generate_ack(task)

    def generate_ack(self, task: Task):
        # Create ACK task with appropriate size and processing demand
        task.data_size = self.config.ack_size
        task.compute_demand = 0
        task.status = TaskStatus.ACK_TRANSMITTING
        #todo set destination
        
        # Set ACK transmission time
        ack_transmission_time = self.estimate_transmission_time(task, task.wd_id)
        task.remaining_times['transmitting'] = ack_transmission_time

    def get_available_resources(self, node_id: int) -> float:
        return self.graph_manager.get_node_features()[node_id][0]

    def get_next_hop(self, task: Task) -> int:
        #todo check if is ask or not!
        path = task.path_for
        if task.current_location in path:
            index = path.index(task.current_location)
            if index + 1 < len(path):
                return path[index + 1]
        return task.destination

    def estimate_transmission_time(self, task: Task, next_node: int) -> float:
        edge_id = self.graph_manager.edge_ids(task.current_location, next_node)
        edge_capacity = self.graph_manager.get_edge_features()[edge_id]
        return task.data_size / edge_capacity if edge_capacity > 0 else float('inf')

    def reset(self):
        self.tasks.clear()
        self.queues.clear()
        self.wireless_device.populate()
        self.graph_manager.update_edge_rates(self.wireless_device)

    def load_sample_tasks(self):
        """
        Load a single sample task into the environment with predefined parameters.
        """
        # Clear existing tasks
        self.tasks.clear()
        
        # Get first available wireless device and server
        wd_ids = list(self.wireless_device.vehicles.keys())
        server_ids = list(self.graph_manager.servers.keys())
        
        if not wd_ids or not server_ids:
            raise ValueError("No wireless devices or servers available")
            
        wd_id = wd_ids[0]
        server_id = server_ids[0]
        
        # Create a single task with predefined parameters
        task = Task(
            task_id=0,
            wd_id=wd_id,
            data_size=5.0,  # 5 MB
            compute_demand=2000,  # 2000 CPU cycles
            deadline=20.0,  # 20 seconds
            current_location=wd_id
        )
        
        # Set destination and path
        task.destination = server_id
        task.path_for = [wd_id, server_id]  # Simple direct path
        task.path_back = [server_id, wd_id]  # Return path
        
        # Set task status and initial transmission time
        task.status = TaskStatus.TRANSMITTING
        task.remaining_times['transmission'] = 10  # 10 seconds for transmission
        
        # Add task to environment
        self.tasks.append(task)
        
        return self.tasks




        ###### important non-implemented issues :

        # transmiting to ap
        #server
        # ...
        # fp
        # retry mechanism if the path corrupted !
        # always check where to stop step
