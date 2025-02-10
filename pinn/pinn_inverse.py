'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 11:26:26
LastEditTime: 2025-02-10 17:03:19
'''
import torch
from pinn.pinn_base import _PINN


class PINNInverse(_PINN):
    def __init__(self, network_solution, network_parameter, should_normalize=True):
        super(PINNInverse, self).__init__()
        self.network_solution = network_solution
        self.network_parameter = network_parameter
        self.should_normalize = should_normalize
        self.mean = None
        self.std = None
        
        self.to(self.device)
        
    def forward(self, data_dict):
        NotImplementedError

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
    
    def net_param(self, X, column_index=None):
        # 当 column_index 为 None 时，X 是完整的
        # 当 column_index 不为 None 时，X 是部分指定列的
        # 例如 net_param(t, column_index=-1) 表示只用到时间 t 预测参数
        
        # 判断 X 的类型，支持 Tensor 和 list/tuple
        # 为了直接输入网络，所以要确保 X 是整体的
        if isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, (list, tuple)):
            X = self.cat_columns(X)
        else:
            raise

        # 标准化
        # 如果没有指定部分列，那么照常标准化
        # 如果指定了部分列，那么就只抽取那些列对应的参数做标准化
        if self.should_normalize:
            if column_index is None:
                X = (X - self.mean) / self.std
            else:
                X = (X - self.mean[column_index]) / self.std[column_index]

        parameter = self.network_parameter(X)

        # 以单列的输出形式给到 output_transform 函数
        X = self.split_columns(X)
        parameter = self.split_columns(parameter)

        # 输出变换
        parameter = self.net_param_output_transform(X, parameter)

        # 确保以单列的输出形式返回
        if isinstance(parameter, torch.Tensor):
            parameter = self.split_columns(parameter)
        elif isinstance(parameter, (list, tuple)):
            pass
        else:
            raise

        return parameter
    
    def net_param_output_transform(self, X, parameter):
        # 需要用户自行定义，默认无变换
        return parameter
    
    def net_res(self, X):
        NotImplementedError

    def net_obs(self, X):
        NotImplementedError
