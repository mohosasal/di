from typing import List

from di.graph.networkx_graph import NetworkXManager
from di.environment.wireless_device import WirelessDevice
from di.environment.task import Task, TaskStatus
from di.environment.mec_environment import MECEnvironment
from exploration.fixed_k_exploration import FixedKExploration
from configs.config import Config


def assign_paths_and_start(env: MECEnvironment,
                           tasks: List[Task],
                           explorer: FixedKExploration):

    path_for, path_back = explorer.explore(tasks, env.graph_manager)

    for task, forward, backward in zip(tasks, path_for, path_back):

        task.path_for = forward
        task.path_back = backward
        task.destination = forward[-1]

        task.current_location = task.wd_id
        task.status = TaskStatus.TRANSMITTING

        if len(forward) > 1:
            next_hop = forward[1]
        else:
            next_hop = forward[0]

        task.remaining_times['transmission'] = env.estimate_transmission_time(task, next_hop)
        env.tasks.append(task)


def main():
    config = Config()
    WirelessDevice.load_data()
    WirelessDevice.move_all(time=1.0)

    graph_manager = NetworkXManager(threshold=config.comm_range)
    for wd in WirelessDevice.get_all():
        graph_manager.add_device(wd)

    env = MECEnvironment(config, graph_manager, WirelessDevice)
    tasks = Task.init_tasks_from_csv(config.task_csv_path)

    explorer = FixedKExploration(
        k=config.max_hops,
    )

    assign_paths_and_start(env, tasks, explorer)

    sim_time = 0.0
    while env.tasks:
        dt = env.step()
        sim_time += dt

    print(f"All tasks completed in {sim_time:.3f}s.")


if __name__ == "__main__":
    main()
