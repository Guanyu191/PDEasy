from typing import List, Tuple, Union

import torch
from pdeasy.framework.framework_base import _NN


class NNForward(_NN):
    def __init__(
            self, 
            network_solution: torch.nn.Module, 
            should_normalize: bool = True
    ):
        r"""Initialize an instance of the NNForward class.

        This class represents a forward neural network model for approximating solutions.
        It allows for optional normalization of input and output data.

        Args:
            network_solution (torch.nn.Module): 
                A neural network module used to approximate the solution.
            should_normalize (bool, optional): 
                A boolean flag indicating whether to enable automatic normalization of input and output data. Defaults to True.

        Attributes:
            network_solution (torch.nn.Module): The neural network for solution approximation.
            should_normalize (bool): Flag indicating whether normalization is enabled.
            X_mean (torch.Tensor): Buffer to store the mean of input coordinates for normalization.
            X_std (torch.Tensor): Buffer to store the standard deviation of input coordinates for normalization.
            U_mean (torch.Tensor): Buffer to store the mean of solution outputs for denormalization.
            U_std (torch.Tensor): Buffer to store the standard deviation of solution outputs for denormalization.
        """
        super(NNForward, self).__init__()
        self.network_solution = network_solution
        self.should_normalize = should_normalize

        self.register_buffer('X_mean', None)
        self.register_buffer('X_std', None)
        self.register_buffer('U_mean', None)
        self.register_buffer('U_std', None)
        
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

        # Concatenate output solution for denormalizing.
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
