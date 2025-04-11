import torch
import torch.nn as nn
import numpy as np
from typing import List
from di.models.actor.base_actor import GNNActor
from di.models.critic.critic import Critic
from di.exploration.base_exploration import ExplorationStrategy
from di.environment.mec_environment import MECEnvironment

class Trainer:
    def __init__(self, actor: GNNActor, critic: Critic, exploration: ExplorationStrategy, 
                 env: MECEnvironment, config):
        self.actor = actor
        self.critic = critic
        self.exploration = exploration
        self.env = env
        self.config = config
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=config.training_params['lr'])
        self.memory = []
    
    def train_step(self, timestep: int):
        self.env.generate_tasks()
        tasks = self.env.tasks
        
        self.actor.eval()
        with torch.no_grad():
            predicted_actions = self.actor(self.env.graph_manager, tasks)
        
        explored_actions = self.exploration.explore(predicted_actions, self.env.graph_manager, 
                                                   self.config.mec_params['max_hops'])
        explored_actions.append([[i] if p < 0.5 else [i, self.config.mec_params['num_wds']] 
                                 for i, p in enumerate(predicted_actions)])
        
        latencies = [self.critic.evaluate(self.env.graph_manager, tasks, action, self.env.queues) 
                     for action in explored_actions]
        best_action = explored_actions[np.argmin(latencies)]
        best_latency = min(latencies)
        
        self.env.step(best_action)
        
        self.memory.append((self.env.graph_manager, tasks[:], best_action, best_latency))
        if len(self.memory) > self.config.training_params['memory_size']:
            self.memory.pop(0)
        
        if timestep % self.config.training_params['delta'] == 0 and self.memory:
            self.actor.train()
            self.optimizer.zero_grad()
            batch = self.memory[np.random.randint(len(self.memory))]
            pred = self.actor(batch[0], batch[1])
            target = torch.tensor([0 if len(path) == 1 else 1 for path in batch[2]], 
                                 dtype=torch.float32)
            loss = nn.BCELoss()(pred, target)
            loss.backward()
            self.optimizer.step()
            print(f"Timestep {timestep}, Loss: {loss.item():.4f}, Latency: {best_latency:.4f}")