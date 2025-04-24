from typing import List, Tuple, Union

import torch
from pdeasy.framework.framework_base import _NN


class DeepONetForward(_NN):
    def __init__(
            self, 
            network_branch: torch.nn.Module, 
            network_trunk: torch.nn.Module, 
            activation_trunk: torch.nn.Module = torch.nn.Tanh(),
            num_outputs: int = 1,
            should_normalize: bool = True
    ):
        r"""Initialize a DeepONetForward model.

        This class represents the forward pass of a DeepONet architecture. A DeepONet consists of 
        a branch network and a trunk network. The branch network processes the input functions, 
        and the trunk network processes the input coordinates. Their outputs are combined to 
        approximate the solution of a function. Input and output normalization can be optionally applied.

        Args:
            network_branch (torch.nn.Module): The branch network module that maps input functions 
                to a feature space.
            network_trunk (torch.nn.Module): The trunk network module that maps input coordinates 
                to a feature space.
            activation_trunk (torch.nn.Module, optional): The activation function applied to the 
                output of the trunk network. Defaults to `torch.nn.ReLU()`.
            num_outputs (int, optional): The number of output dimensions of the model. Defaults to 1.
            should_normalize (bool, optional): A boolean flag indicating whether to normalize the 
                input functions, input coordinates, and output solutions. Defaults to True.

        Attributes:
            network_branch (torch.nn.Module): The branch network module.
            network_trunk (torch.nn.Module): The trunk network module.
            activation_trunk (torch.nn.Module): The activation function for the trunk network.
            should_normalize (bool): Flag to control input and output normalization.
            bias_last (torch.nn.Parameter): A learnable bias parameter for the final output layer.
            F_mean (torch.Tensor): Buffer for the mean of the branch network inputs for normalization.
            F_std (torch.Tensor): Buffer for the standard deviation of the branch network inputs for normalization.
            X_mean (torch.Tensor): Buffer for the mean of the trunk network inputs for normalization.
            X_std (torch.Tensor): Buffer for the standard deviation of the trunk network inputs for normalization.
            U_mean (torch.Tensor): Buffer for the mean of the output solutions for denormalization.
            U_std (torch.Tensor): Buffer for the standard deviation of the output solutions for denormalization.
        """
        super(DeepONetForward, self).__init__()
        self.network_branch = network_branch
        self.network_trunk = network_trunk
        self.activation_trunk = activation_trunk
        self.should_normalize = should_normalize
        self.bias_last = torch.nn.Parameter(torch.zeros(num_outputs))

        # F: input of branch network.
        # X: input of trunk network.
        # U: output of solution for function F at location X.
        # Solve equation like 'dU/dX = F'.
        self.register_buffer('F_mean', None)
        self.register_buffer('F_std', None)
        self.register_buffer('X_mean', None)
        self.register_buffer('X_std', None)
        self.register_buffer('U_mean', None)
        self.register_buffer('U_std', None)
        
        self.to(self.device)

    def net_branch(
            self, 
            F: torch.Tensor
    ) -> torch.Tensor:
        
        # Normalize input functions.
        if (
            self.should_normalize and
            self.F_mean is not None and
            self.F_std is not None
        ):
            F = (F - self.F_mean) / self.F_std

        branch = self.network_branch(F)
        return branch
    
    def net_trunk(
            self,
            X: torch.Tensor
    ) -> torch.Tensor:
        
        # Normalize input coordinates.
        if (
            self.should_normalize and
            self.X_mean is not None and
            self.X_std is not None 
        ):
            X = (X - self.X_mean) / self.X_std

        trunk = self.network_trunk(X)
        trunk = self.activation_trunk(trunk)
        return trunk

    def net_sol(
            self, 
            F: torch.Tensor,
            X: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
        ) -> Union[torch.Tensor, List[torch.Tensor]]:

        # Only support tensor, list or tuple of tensor.
        if isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, (list, tuple)):
            X = self.cat_columns(X)
        else:
            raise
                
        branch = self.net_branch(F)  # (n_func, n_outdim, n_p)
        trunk = self.net_trunk(X)  # (n_func, n_p)
        trunk = trunk.unsqueeze(1)  # (n_func, 1, n_p)

        solution = torch.sum(branch * trunk, dim=-1, keepdim=False) + self.bias_last

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
