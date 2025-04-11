class Config:
    def __init__(self):
        # MEC environment parameters
        self.mec_params = {
            'num_wds': 3, 'num_aps': 2, 'num_servers': 1,
            'd_i_range': [0.256, 4], 'k_i_range': [3, 150],
            'f_wd_range': [150, 1000], 'f_ap': 500, 'f_server': 10000,
            'bandwidth': 10, 'transmit_power': 1, 'noise': 1e-9,
            'max_hops': 2, 'mobility_speed': 5.0, 'grid_size': 100,
            'vehicle_computing_capacity': 500.0,  # Computing capacity of vehicles
            'ap_computing_capacity': 2000.0,      # Computing capacity of access points
            'server_computing_capacity': 10000.0, # Computing capacity of servers
        }
        self.training_params = {
            'lr': 0.001, 'delta': 5, 'memory_size': 256, 'k_explore': 3,
            'batch_size': 32,
            'target_update_freq': 100,
            'gamma': 0.99,
        }
        self.ack_size = 0.1  # Size of acknowledgment packets (MB)
        self.node_feature_dim = 3  # Number of features per node
        self.wd_to_wd_threshold = 1