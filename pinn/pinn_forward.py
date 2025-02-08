'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 11:26:26
LastEditTime: 2025-02-08 21:50:10
'''
import torch
from pinn.pinn_base import _PINN


class PINNForward(_PINN):
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
        # 判断 X 的类型
        if isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, (list, tuple)):
            X = torch.cat(X, dim=1)
        else:
            raise ValueError(f"Unsupported type of X: {type(X)}")
        # 标准化
        if self.should_normalize:
            X = (X - self.mean) / self.std
        solution = self.network_solution(X)
        return solution
    
    def init_net_res_input(self, X):
        num_columns = X.size(1)
        if num_columns > 1:
            columns = []
            for i in range(num_columns):
                column = X[:, [i]]
                column.requires_grad_(True)
                columns.append(column)
            return columns
        elif num_columns == 1:
            X.requires_grad_(True)
            return X
    
    def net_res(self, X):
        NotImplementedError

    def net_bcs(self, X):
        NotImplementedError

    def net_ics(self, X):
        NotImplementedError
