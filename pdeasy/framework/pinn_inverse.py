from typing import List, Tuple, Union

import torch
from pdeasy.framework.framework_base import _PINN


class PINNInverse(_PINN):
    def __init__(
            self, 
            network_solution: torch.nn.Module, 
            network_parameter: torch.nn.Module, 
            should_normalize: bool = True
        ):
        r"""Initialize an inverse Physics-Informed Neural Network (PINN) model.

        This inverse PINN model is designed to estimate unknown parameters in a physical system.
        It takes two neural networks as inputs: one for approximating the solution and another
        for estimating the parameters. Additionally, it supports optional normalization of
        input and output data.

        Args:
            network_solution (torch.nn.Module): 
                A PyTorch neural network module used to approximate the solution of the physical system.
            network_parameter (torch.nn.Module): 
                A PyTorch neural network module used to estimate the unknown parameters of the physical system.
            should_normalize (bool, optional): 
                A boolean flag indicating whether to enable automatic normalization of input and output data.
                If set to True, the model will normalize the input coordinates and denormalize the output
                solution and parameters. Defaults to True.

        Attributes:
            network_solution (torch.nn.Module): The neural network for solution approximation.
            network_parameter (torch.nn.Module): The neural network for parameter estimation.
            should_normalize (bool): Flag indicating whether normalization is enabled.
            X_mean (torch.Tensor): Buffer to store the mean of input coordinates for normalization.
            X_std (torch.Tensor): Buffer to store the standard deviation of input coordinates for normalization.
            U_mean (torch.Tensor): Buffer to store the mean of solution outputs for denormalization.
            U_std (torch.Tensor): Buffer to store the standard deviation of solution outputs for denormalization.
            P_mean (torch.Tensor): Buffer to store the mean of parameter estimates for denormalization.
            P_std (torch.Tensor): Buffer to store the standard deviation of parameter estimates for denormalization.
        """
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
