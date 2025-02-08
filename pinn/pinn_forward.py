'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 11:26:26
LastEditTime: 2025-02-08 14:18:09
'''
from pinn.pinn_base import PINNBase


class PINNForward(PINNBase):
    def __init__(self, network_solution, should_normalize=True):
        super(PINNForward, self).__init__()
        self.network_solution = network_solution
        self.should_normalize = should_normalize
        self.mean = None
        self.std = None
        
        self.to(self.device)
        
    def forward(self, data_dict):
        NotImplementedError

    def net_sol(self, X):
        if self.should_normalize:
            X = (X - self.mean) / self.std
        solution = self.network_solution(X)
        return solution
    
    def net_res(self, X):
        NotImplementedError

    def net_bcs(self, X):
        NotImplementedError

    def net_ics(self, X):
        NotImplementedError
