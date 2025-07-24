import csv
from enum import Enum, auto
from typing import List, Set, Optional, Dict


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

        # Graph-related attributes
        self.current_location = current_location
        self.destination = None
        self.path_for = None
        self.path_back = None

        # DAG-related attributes
        self.dependencies: Set['Task'] = set()
        self.dependents: Set['Task'] = set()

    def add_dependency(self, dependency: 'Task'):
        self.dependencies.add(dependency)
        dependency.dependents.add(self)

    def is_ready(self):
        return all(dep.status == TaskStatus.FULL_COMPLETED for dep in self.dependencies)

    def init_tasks_from_csv(csv_path: str) -> List['Task']:
        tasks: Dict[int, Task] = {}
        dependencies_map: Dict[int, List[int]] = {}

        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                task_id = int(row['task_id'])
                wd_id = row['wd_id']
                data_size = float(row['data_size'])
                compute_demand = float(row['compute_demand'])
                deadline = float(row['deadline'])
                current_location = int(row['current_location'])
                deps = [int(d) for d in row['dependencies'].split('|') if d.strip()]

                task = Task(task_id, wd_id, data_size, compute_demand, deadline, current_location)
                tasks[task_id] = task
                dependencies_map[task_id] = deps

        for task_id, dep_ids in dependencies_map.items():
            for dep_id in dep_ids:
                tasks[task_id].add_dependency(tasks[dep_id])

        return list(tasks.values())




    # a task only should be computeld only if its parrtent task are complete and acked... so in this way how do we can do this ?
    # 1 - a cluster of tasks should be computed only in one host all togathere
    # 2 - a cluster tasks could be in any car and then how they should contact to each other ?
    # for now we choode option 1


    # gang task should be handled in multicore in edge server
    # this should be handled
    # evolving gnn for dynamic task dags

    # output of migration numbers deadline misses and ...

    # handle offloading if failed not discarding the task


    # todo handle dags with networkX

    # todo handle computation