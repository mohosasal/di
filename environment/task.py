class Task:
    def __init__(self, d_i: float, k_i: float, gamma_i: float, tau_i: float, wd_id: int):
        self.d_i = d_i
        self.k_i = k_i
        self.gamma_i = gamma_i
        self.tau_i = tau_i
        self.wd_id = wd_id

        # update the parameters