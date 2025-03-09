from typing import List, Tuple, Union

import torch
from framework.framework_base import _PINN


class PIDeepONetForward(_PINN):
    def __init__(
            self, 
            network_branch: torch.nn.Module, 
            network_trunk: torch.nn.Module, 
            activation_trunk: torch.nn.Module = torch.tanh,
            num_outputs: int = 1,
            should_normalize: bool = True
    ):
        r"""Initialize an instance of the PIDeepONetForward class.

        This class represents a forward model of the Physics-Informed Deep Operator Network.
        It combines a branch network and a trunk network to approximate solutions of a physical system.

        Args:
            network_branch (torch.nn.Module): 
                The branch network that processes the input functions.
            network_trunk (torch.nn.Module): 
                The trunk network that processes the input coordinates.
            activation_trunk (torch.nn.Module, optional): 
                The activation function for the trunk network. Defaults to torch.tanh.
            num_outputs (int, optional): 
                The number of outputs of the model. Defaults to 1.
            should_normalize (bool, optional): 
                Whether to normalize the inputs and outputs. Defaults to True.

        Attributes:
            network_branch (torch.nn.Module): The branch network.
            network_trunk (torch.nn.Module): The trunk network.
            activation_trunk (torch.nn.Module): The activation function for the trunk network.
            should_normalize (bool): Flag indicating whether normalization should be applied.
            bias_last (torch.nn.Parameter): The bias parameter for the last layer.
            F_mean (torch.Tensor): Buffer for the mean of the branch network inputs.
            F_std (torch.Tensor): Buffer for the standard deviation of the branch network inputs.
            X_mean (torch.Tensor): Buffer for the mean of the trunk network inputs.
            X_std (torch.Tensor): Buffer for the standard deviation of the trunk network inputs.
            U_mean (torch.Tensor): Buffer for the mean of the solution outputs.
            U_std (torch.Tensor): Buffer for the standard deviation of the solution outputs.
        """
        super(PIDeepONetForward, self).__init__()
        self.network_branch = network_branch
        self.network_trunk = network_trunk
        self.activation_trunk = activation_trunk
        self.should_normalize = should_normalize
        self.bias_last = torch.nn.Parameter(torch.zeros(num_outputs))

        # F: input of branch network.
        # X: input of trunk network.
        # U: output of solution for function F at location X.
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
                
        branch = self.net_branch(F)
        trunk = self.net_trunk(X)

        solution = torch.sum(branch * trunk, dim=-1, keepdim=True) + self.bias_last

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

    def net_res(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def net_bcs(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def net_ics(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError