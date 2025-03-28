class Config:
    def __init__(self):
        self.mec_params = {
            'num_wds': 3, 'num_aps': 2, 'num_servers': 1,
            'd_i_range': [0.256, 4], 'k_i_range': [3, 150],
            'f_wd_range': [150, 1000], 'f_ap': 500, 'f_server': 10000,
            'bandwidth': 10, 'transmit_power': 1, 'noise': 1e-9,
            'max_hops': 2, 'mobility_speed': 5.0, 'grid_size': 100
        }
        self.training_params = {
            'lr': 0.001, 'delta': 5, 'memory_size': 256, 'k_explore': 3
        }