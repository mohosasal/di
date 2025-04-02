from configs.config import Config
from graph.dgl_graph import GraphManager
from environment.mobility.random_mobility import RandomMobility
from environment.mec_environment import MECEnvironment
from models.actor.gcn_actor import GCNActor
from models.critic.critic import Critic
from exploration.fixed_k_exploration import FixedKExploration
from trainer.trainer import Trainer
from environment.wireless_device import *

def main():
    config = Config()
    wd = WirelessDevice()
    wd.load_data()
    wd.populate()
    wd.move_all()
    print(wd)

    graph_manager = GraphManager(config)
    mobility_model = RandomMobility(config)
    env = MECEnvironment(config, graph_manager, mobility_model)
    actor = GCNActor(in_feats=3, hidden_feats=16)
    critic = Critic()
    exploration = FixedKExploration(config.training_params['k_explore'])
    trainer = Trainer(actor, critic, exploration, env, config)

    for t in range(100):
        trainer.train_step(t)

if __name__ == "__main__":
    main()