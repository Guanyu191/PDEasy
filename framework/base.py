from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn


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
    def __init__(self):
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
        return _grad(outputs, inputs, n_order=n_order)
    
    @staticmethod
    def split_columns_and_requires_grad(
        X:torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        return _split_columns_and_requires_grad(X)
