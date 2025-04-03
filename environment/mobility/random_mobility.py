import numpy as np
from .base_wd import MobilityModel
from typing import Dict, Tuple

class RandomMobility(MobilityModel):
    def __init__(self, config):
        self.config = config
        self.velocities = {i: (np.random.uniform(-1, 1), np.random.uniform(-1, 1)) 
                          for i in range(config.mec_params['num_wds'])}
    
    def move(self, positions: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
        speed = self.config.mec_params['mobility_speed']
        grid_size = self.config.mec_params['grid_size']
        new_positions = positions.copy()
        
        for wd in range(self.config.mec_params['num_wds']):
            vx, vy = self.velocities[wd]
            dx, dy = vx * speed, vy * speed
            x, y = new_positions[wd]
            new_x = max(0, min(grid_size, x + dx))
            new_y = max(0, min(grid_size, y + dy))
            new_positions[wd] = (new_x, new_y)
            # Randomly adjust direction occasionally
            if np.random.random() < 0.1:
                self.velocities[wd] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        
        return new_positions