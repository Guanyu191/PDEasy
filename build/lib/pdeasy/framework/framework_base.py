from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn

__all__ = [
    "_NN",
    "_PINN", 
]


def _split_columns(
        X:torch.Tensor
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(X, torch.Tensor) and X.dim() == 2:
        num_columns = X.shape[1]
        if num_columns == 1:
            return X
        else:
            return [X[:, [i]] for i in range(num_columns)]
    else:
        raise


def _cat_columns(
        X:Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
) -> torch.Tensor:
    
    if isinstance(X, (list, tuple)) and X[0].ndim == 2:
        return torch.cat(X, dim=1)
    elif isinstance(X, torch.Tensor) and X.ndim == 2:
        return X
    else:
        raise


def _split_columns_and_requires_grad(
        X:torch.Tensor
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(X, torch.Tensor) and X.dim() == 2:
        num_columns = X.shape[1]
        if num_columns == 1:
            return X.requires_grad_(True)
        else:
            return [X[:, [i]].requires_grad_(True) for i in range(num_columns)]
    else:
        raise


def _torch_grad(
        outputs, 
        inputs
):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)[0]


def _grad(
        outputs:torch.Tensor, 
        inputs:torch.Tensor, 
        n_order:int = 1
) -> torch.Tensor:
    
    current_grads = outputs
    for k in range(1, n_order + 1):
        if k == 1:
            current_grads = _torch_grad(outputs, inputs)
        else:
            current_grads = _torch_grad(current_grads, inputs)
    return current_grads


class _NN(nn.Module):
    r"""The base class of the data-driven class models.
    """
    def __init__(self):
        r"""Initialize the _NN base class.

        This class serves as a base for neural network models. It sets up the device
        to be used for computations, either CUDA if available or CPU.

        Attributes:
            device (torch.device): The device on which the model will run.
        """
        super(_NN, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def forward(
            self, 
            X:torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
    
    @staticmethod
    def split_columns(
        X:torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        return _split_columns(X)
    
    @staticmethod
    def cat_columns(
        X_list:List[torch.Tensor]
    ) -> torch.Tensor:
        return _cat_columns(X_list)


class _PINN(_NN):
    r"""The base class of the physically driven class models.
    """
    def __init__(self):
        super(_PINN, self).__init__()

    def forward(
            self, 
            data_dict:Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def grad(
        outputs:torch.Tensor, 
        inputs:torch.Tensor, 
        n_order:int = 1
    ) -> torch.Tensor:
        r"""Compute the specified order gradient of the outputs with respect to the inputs.

        This static method calculates the gradient of a given order for the outputs 
        relative to the inputs. It uses the underlying `_grad` function to perform 
        the actual computation.

        Args:
            outputs (torch.Tensor): The output tensor with respect to which the gradient is calculated.
            inputs (torch.Tensor): The input tensor for which the gradient is computed.
            n_order (int, optional): The order of the gradient. Defaults to 1.

        Returns:
            torch.Tensor: The computed gradient tensor of the specified order.
        """
        return _grad(outputs, inputs, n_order=n_order)
    
    @staticmethod
    def split_columns_and_requires_grad(
        X:torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        r"""Split the columns of a tensor and set the 'requires_grad' attribute to True.

        This method is a wrapper for the internal '_split_columns_and_requires_grad' function.
        It takes a 2D tensor as input, splits its columns, and ensures that gradient computation
        is enabled for each resulting tensor.

        Args:
            X (torch.Tensor): The input tensor to be split. It should be a 2D tensor.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: If the input tensor has only one column,
            it returns the tensor itself with 'requires_grad' set to True. Otherwise, it returns
            a list of tensors, where each tensor corresponds to a column of the input tensor
            with 'requires_grad' set to True.
        """
        return _split_columns_and_requires_grad(X)
