from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from ..graph.base_graph import GraphInterface
from .mobility.base_mobility import MobilityModel
from .wireless_device import WirelessDevice
from .task import Task


class MECEnvironment:
    def __init__(self, config, graph_manager: GraphInterface, mobility_model: MobilityModel):
        self.config = config
        self.graph_manager = graph_manager
        self.mobility_model = mobility_model
        self.wireless_devices: List[WirelessDevice] = []
        self.tasks: List[Task] = []
        self.queues: Dict[int, List[Task]] = defaultdict(list)
        self.positions: Dict[int, Tuple[float, float]] = self._init_positions()
        self.graph_manager.update_edge_rates(self.positions)

    def _init_positions(self) -> Dict[int, Tuple[float, float]]:
        """Initialize positions for all devices and APs."""
        num_wds = self.config.mec_params['num_wds']
        num_aps = self.config.mec_params['num_aps']
        
        # Initialize wireless devices
        for wd_id in range(num_wds):
            position = (np.random.uniform(0, 100), np.random.uniform(0, 100))
            wd = WirelessDevice(wd_id, position, self.config.mec_params)
            self.wireless_devices.append(wd)
        
        # Initialize AP positions
        positions = {wd.wd_id: wd.position for wd in self.wireless_devices}
        for i in range(num_wds, num_wds + num_aps):
            positions[i] = (50 + 20 * (i - num_wds), 50)
        
        return positions
    
    def _update_positions(self):
        """Update positions of all devices using the mobility model."""
        self.positions = self.mobility_model.move(self.positions)
        for wd in self.wireless_devices:
            wd.update_position(self.positions[wd.wd_id])
        self.graph_manager.update_edge_rates(self.positions)
    
    def _process_tasks(self, actions: List[List[int]]) -> float:
        """Process tasks according to the given actions and calculate total latency."""
        total_latency = 0.0
        node_features = self.graph_manager.get_node_features()
        edge_features = self.graph_manager.get_edge_features()
        
        # Queue tasks according to actions
        for task, path in zip(self.tasks, actions):
            final_node = path[-1] if len(path) > 1 else task.wd_id
            self.queues[final_node].append(task)
        
        # Process queued tasks
        for node_id, queue in self.queues.items():
            for task in queue:
                path = next(p for p, t in zip(actions, self.tasks) if t.wd_id == task.wd_id)
                latency = self._calculate_task_latency(task, path, node_features, edge_features)
                total_latency += latency
                
                # Update queue delay
                node_features[node_id][1] = len(queue) * 0.01
        
        return total_latency
    
    def _calculate_task_latency(self, task: Task, path: List[int], 
                              node_features: List[List[float]], 
                              edge_features: List[float]) -> float:
        """Calculate latency for a single task."""
        if len(path) == 1:  # Local processing
            wd = self.wireless_devices[task.wd_id]
            T_l = wd.process_task(task) + node_features[path[0]][1]
            # Update battery consumption
            energy_consumed = task.k_i / (wd.computing_power * 1000)
            wd.update_battery(energy_consumed)
            return task.gamma_i * T_l
        else:  # Multi-hop
            T_trans = sum(task.d_i / edge_features[self.graph_manager.edge_ids(path[i], path[i + 1])]
                          for i in range(len(path) - 1))
            T_edge = task.k_i / node_features[path[-1]][0]
            return task.gamma_i * (T_trans + T_edge)
    
    def _update_network_topology(self):
        """Randomly update network topology between WDs and APs."""
        if np.random.random() < 0.05:
            wd = np.random.choice(self.wireless_devices)
            ap = np.random.randint(self.config.mec_params['num_wds'], 
                                  self.config.mec_params['num_wds'] + self.config.mec_params['num_aps'])
            if self.graph_manager.has_edges_between(wd.wd_id, ap):
                self.graph_manager.remove_edges(self.graph_manager.edge_ids(wd.wd_id, ap))
            else:
                self.graph_manager.add_edges(wd.wd_id, ap)
                self.graph_manager.update_edge_rates(self.positions)
    
    def generate_tasks(self):
        """Generate new tasks for all wireless devices."""
        self.tasks = [wd.generate_task() for wd in self.wireless_devices]
    
    def step(self, actions: List[List[int]]) -> float:
        """Execute one step of the environment."""
        self.queues.clear()
        
        # Update environment state
        self._update_positions()
        self._update_network_topology()
        
        # Process tasks and calculate latency
        total_latency = self._process_tasks(actions)
        
        return total_latency
    
    def reset(self):
        """Reset the environment to its initial state."""
        self.positions = self._init_positions()
        self.tasks = []
        self.queues.clear()
        if hasattr(self.mobility_model, 'reset'):
            self.mobility_model.reset()
