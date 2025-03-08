from typing import List, Tuple, Union

import torch
from framework.base import _PINN


class PINNInverse(_PINN):
    def __init__(
            self, 
            network_solution: torch.nn.Module, 
            network_parameter: torch.nn.Module, 
            should_normalize: bool = True
        ):

        super(PINNInverse, self).__init__()
        self.network_solution = network_solution
        self.network_parameter = network_parameter
        self.should_normalize = should_normalize

        self.register_buffer('X_mean', None)
        self.register_buffer('X_std', None)
        self.register_buffer('U_mean', None)
        self.register_buffer('U_std', None)
        self.register_buffer('P_mean', None)
        self.register_buffer('P_std', None)
        
        self.to(self.device)

    def net_sol(
            self, 
            X: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
        ) -> Union[torch.Tensor, List[torch.Tensor]]:

        # Only support tensor, list or tuple of tensor.
        if isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, (list, tuple)):
            X = self.cat_columns(X)
        else:
            raise
        
        # Normalize input coordinates.
        if (
            self.should_normalize and
            self.X_mean is not None and
            self.X_std is not None
        ):
            X = (X - self.X_mean) / self.X_std
        
        solution = self.network_solution(X)

        # Split input coordinates and output solution for output transform.
        X = self.split_columns(X)
        solution = self.split_columns(solution)
        solution = self.net_sol_output_transform(X, solution)

        # Cat output solution for denormalizing.
        solution = self.cat_columns(solution)
        if (
            self.should_normalize and
            self.U_mean is not None and
            self.U_std is not None
        ):
            solution = solution * self.U_std + self.U_mean

        # Ensure output solution is a tensor or a list of tensor.
        solution = self.split_columns(solution)
        return solution
    
    def net_sol_output_transform(
            self, 
            X: Union[torch.Tensor, List[torch.Tensor]], 
            solution: Union[torch.Tensor, List[torch.Tensor]]
        ) -> Union[torch.Tensor, List[torch.Tensor]]:
        return solution
    
    def net_param(
            self, 
            X: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]], 
            column_index: int = None
        ) -> Union[torch.Tensor, List[torch.Tensor]]:

        # Only support tensor, list or tuple of tensor.
        if isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, (list, tuple)):
            X = self.cat_columns(X)
        else:
            raise

        # Normalize input coordinates which index is column_index.
        if (
            self.should_normalize and
            self.X_mean is not None and
            self.X_std is not None 
        ):
            if column_index is None:
                X = (X - self.X_mean) / self.X_std
            else:
                X = (X - self.X_mean[:, column_index]) / self.X_std[:, column_index]

        parameter = self.network_parameter(X)

        # Split input coordinates and output parameter for output transform.
        X = self.split_columns(X)
        parameter = self.split_columns(parameter)
        parameter = self.net_param_output_transform(X, parameter)

        # Concatenate output parameter for denormalizing.
        parameter = self.cat_columns(parameter)
        if (
            self.should_normalize and
            self.P_mean is not None and
            self.P_std is not None
        ):
            parameter = parameter * self.P_std + self.P_mean

        # Ensure output parameter is a tensor or a list of tensor.
        parameter = self.split_columns(parameter)
        return parameter
    
    def net_param_output_transform(
        self, 
        X: Union[torch.Tensor, List[torch.Tensor]], 
        parameter: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        return parameter
    
    def net_res(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
