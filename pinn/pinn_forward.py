'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 11:26:26
LastEditTime: 2025-02-12 02:04:57
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
        raise NotImplementedError

    def net_sol(self, X):
        # 判断 X 的类型，支持 Tensor 和 list/tuple
        if isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, (list, tuple)):
            X = self.cat_columns(X)
        else:
            raise
        
        # 标准化
        if self.should_normalize:
            X = (X - self.mean) / self.std
        solution = self.network_solution(X)

        # 以单列的输出形式给到 output_transform 函数
        X = self.split_columns(X)
        solution = self.split_columns(solution)

        # 输出变换
        solution = self.net_sol_output_transform(X, solution)

        # 确保以单列的输出形式返回
        if isinstance(solution, torch.Tensor):
            solution = self.split_columns(solution)
        elif isinstance(solution, (list, tuple)):
            pass
        else:
            raise
        return solution
    
    def net_sol_output_transform(self, X, solution):
        # 需要用户自行定义，默认无变换
        return solution
        
    def net_res(self, X):
        raise NotImplementedError

    def net_bcs(self, X):
        raise NotImplementedError

    def net_ics(self, X):
        raise NotImplementedError
