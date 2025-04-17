from di.graph.networkx_graph import *
from models.actor.pyg_actor import PyGActor
from environment.mec_environment import MECEnvironment
from environment.wireless_device import WirelessDevice
from models.critic.critic import Critic
from exploration.fixed_k_exploration import FixedKExploration
#from training.trainer import Trainer
from configs.config import Config

def main():
    config = Config()

    WirelessDevice.load_data()
    WirelessDevice.move_all(1)
    graph_manager = NetworkXManager(threshold=1500)
    [graph_manager.add_device(i) for i in WirelessDevice.vehicles]


    env = MECEnvironment(config, graph_manager,WirelessDevice)
    env.load_sample_tasks()


    env.step()
    env.step()
    env.step()
    env.step()

    temporal_data = graph_manager.get_temporal_data()


if __name__ == "__main__":
    main()