from di.graph.networkx_graph import *
from models.actor.pyg_actor import PyGActor
from environment.mec_environment import MECEnvironment
from environment.wireless_device import WirelessDevice
from models.critic.critic import Critic
from exploration.fixed_k_exploration import FixedKExploration
#from training.trainer import Trainer
from configs.config import Config

def main():
    # Initialize components
    config = Config()
    
    # Create graph manager
    graph_manager = NetworkXManager(threshold=10)
    
    # Create wireless device manager
    WirelessDevice.load_data()
    WirelessDevice.move_all(0)
    


    #
    # # Create environment
    env = MECEnvironment(config, graph_manager)
    env.step()
    env.step()
    #
    # # Create actor and critic
    # actor = PyGActor(in_feats=config.node_feature_dim, hidden_feats=64)
    # critic = Critic()
    # exploration = FixedKExploration(config.training_params['k_explore'])
    # trainer = Trainer(actor, critic, exploration, env, config)
    #
    # env.load_sample_tasks()
    # env.step()
    #
    # for t in range(100):
    #     trainer.train_step(t)

if __name__ == "__main__":
    main()