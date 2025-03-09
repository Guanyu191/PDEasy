r"""Fourier Feature Network type neural network models.

The Fourier Feature type neural network models are referenced from
https://www.sciencedirect.com/science/article/pii/S0045782521002759
It includes 3 types of models:
    1. Multiscale Fourier Feature Model (MFF).
    2. Spatio Temporal Fourier Feature Model (STFF).
    3. Spatio Temporal Multiscale Fourier Feature Model (STMFF).

Specifically, we implement FFN models for 1D, 2D, 1DT, and 2DT problems, including:
    1. MFF1D: Multiscale Fourier Feature Model for 1D Space.
    2. STFF1DT: Spatio Temporal Fourier Feature Model for 1D Space and Time.
    3. STMFF1DT: Spatio Temporal Multiscale Fourier Feature Model for 1D Space and Time.
    4. FF2D: Fourier Feature Model for 2D Space.
    5. MFF2D: Multiscale Fourier Feature Model for 2D Space.
    6. FF2DT: Fourier Feature Model for 2D Space and Time.

To create a network instance, simply pass in a list representing the number of layers 
and the number of neurons in each layer of the neural network.
In addition, you can adjust the activation function of the network and the method 
for initializing network parameters.

The default activation function of the network is Tanh. Other options include:
    1. ReLU
    2. LeakyReLU
    3. Tanh
    4. Sigmoid
    5. GELU
    6. SELU
    7. Softplus
    8. Hardtanh
    9. PReLU
    10. RReLU
    11. ELU

The default method for initializing network parameters is "xavier_normal". Other options include:
    1. kaiming_normal
    2. kaiming_uniform
    3. xavier_normal
    4. xavier_uniform
    5. normal
    6. uniform
    7. constant
    8. default

Example::
    >>> NN_LAYERS = [1, 100, 100, 1]
    >>> network = MFF1D(NN_LAYERS)
"""
import torch
import torch.nn as nn

from utils.init_network_weights import init_network_weights
from utils.init_network_activation_function import init_network_activation_function

from typing import List, Tuple, Union
from torch import Tensor


class MFF1D(nn.Module):
    r"""Multiscale Fourier Feature Model for 1D Space.

    Args:
        nn_layers (List[int]): The layer structure of the neural network.
            For example, [1, 100, 100, 100, 1].
            Note that nn_layers[1] must be an even number.
        sigma_1 (int, optional): The parameter of Fourier features.
            The first scale is generally set to 1, ...
            The larger sigma_1 is, the higher the overall frequency of the learned spatio domain.
        sigma_2 (int, optional): The parameter of Fourier features.
            The second scale is generally set to 10, 20, 50, 100, 200, ...
            The larger sigma_2 is, the higher the overall frequency of the learned spatio domain.

    Note::
        1. nn_layers[0] is the original [x] input.
        2. [x] is transformed into H_1 and H_2 through Fourier Embedding.
        3. Among them, W_1 and W_2 are the parameters of Fourier Embedding.
        4. Then H_1 and H_2 are successively fed into the same network (e.g., MLP).
        5. Finally, H = [H_1, H_2] is passed as the concatenated feature to the last linear layer to return solution.

    Example::
        >>> NN_LAYERS = [1, 100, 100, 100, 1]
        >>> network = MFF1D(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh',
            init_type: str = 'xavier_normal',
            sigma_1: int = 1, 
            sigma_2: int = 10,
    ):
        r"""Initialize the MFF1D model.

        Args:
            nn_layers (List[int]): The layer structure of the neural network, e.g., [1, 100, 100, 100, 1]. 
                Note that nn_layers[1] must be an even number.
            act_type (str, optional): The type of activation function. Defaults to 'tanh'.
            init_type (str, optional): The method for initializing network parameters. Defaults to 'xavier_normal'.
            sigma_1 (int, optional): The first scale parameter for Fourier features. Defaults to 1.
            sigma_2 (int, optional): The second scale parameter for Fourier features. Defaults to 10.

        Attributes:
            args (dict): A dictionary storing all input arguments for reference.
            sigma_1 (int): The first scale parameter for Fourier features.
            sigma_2 (int): The second scale parameter for Fourier features.
            W_1 (nn.Parameter): Fourier feature matrix for the first scale, non - trainable.
            W_2 (nn.Parameter): Fourier feature matrix for the second scale, non - trainable.
            model_ (nn.Sequential): A sequential container representing the neural network model excluding the last layer.
            last_layer_ (nn.Linear): The final linear layer of the network.
        """
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
            'sigma_1': sigma_1,
            'sigma_2': sigma_2
        }
        super(MFF1D, self).__init__()

        # Initialize the Fourier features.
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.W_1 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_1, requires_grad=False)
        self.W_2 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_2, requires_grad=False)
        
        # Initialize the backbone network.
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i+1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i+1}', init_network_activation_function(act_type))

        self.last_layer_ = nn.Linear(2 * nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        H_1 = torch.cat([torch.cos(torch.matmul(X, self.W_1)), 
                         torch.sin(torch.matmul(X, self.W_1))], dim=1)
        H_2 = torch.cat([torch.cos(torch.matmul(X, self.W_2)), 
                         torch.sin(torch.matmul(X, self.W_2))], dim=1)
        
        H_1 = self.model_(H_1)
        H_2 = self.model_(H_2)

        H = torch.cat([H_1, H_2], dim=1)
        u = self.last_layer_(H)
        return u


class STFF1DT(nn.Module):
    r"""Spatio Temporal Fourier Feature Model for 1D Space and Time.

    Args:
        nn_layers: The layer structure of the neural network.
            For example, [2, 100, 100, 100, 1].
            Note that nn_layers[1] must be an even number.
        sigma_x: The parameter of Fourier features for the spatial domain.
            Generally, it takes values like 1, 20, 50, 100, 200, ...
            The larger sigma_x is, the higher the frequency of the learned spatial domain.
        sigma_t: The parameter of Fourier features for the temporal domain.
            Generally, it takes values like 1, ...
            The larger sigma_t is, the higher the frequency of the learned temporal domain.

    Note::
        1. nn_layers[0] is the original [x, t] input.
        2. [x, t] is transformed into H_x and H_t through Fourier Embedding.
        3. Among them, W_x and W_t are the parameters of Fourier Embedding.
        4. Then H_x and H_t are successively fed into the same network (e.g., MLP).
        5. Finally, H = H_x * H_t is passed as the fused feature to the last linear layer to return solution.

    Example::
        >>> NN_LAYERS = [2, 100, 100, 100, 1]
        >>> network = STFF1DT(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh',
            init_type: str = 'xavier_normal',
            sigma_x: int = 1, 
            sigma_t: int = 10
    ):
        r"""
        Initialize the STFF1DT model.

        Args:
            nn_layers (List[int]): The layer structure of the neural network, e.g., [2, 100, 100, 100, 1]. 
                Note that nn_layers[1] must be an even number.
            act_type (str, optional): The type of activation function. Defaults to 'tanh'.
            init_type (str, optional): The method for initializing network parameters. Defaults to 'xavier_normal'.
            sigma_x (int, optional): The parameter of Fourier features for the spatial domain. Defaults to 1.
            sigma_t (int, optional): The parameter of Fourier features for the temporal domain. Defaults to 10.

        Attributes:
            args (dict): A dictionary storing all input arguments for reference.
            sigma_x (int): The parameter of Fourier features for the spatial domain.
            sigma_t (int): The parameter of Fourier features for the temporal domain.
            W_x (nn.Parameter): Fourier feature matrix for the spatial domain, non - trainable.
            W_t (nn.Parameter): Fourier feature matrix for the temporal domain, non - trainable.
            model_ (nn.Sequential): A sequential container representing the neural network model excluding the last layer.
            last_layer_ (nn.Linear): The final linear layer of the network.
        """
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
            'sigma_x': sigma_x,
            'sigma_t': sigma_t
        }
        super(STFF1DT, self).__init__()

        self.sigma_x = sigma_x
        self.sigma_t = sigma_t
        self.W_x = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_t = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_t, requires_grad=False)
        
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, t = X[:, [0]], X[:, [1]]

        H_x = torch.cat([torch.cos(torch.matmul(x, self.W_x)), 
                         torch.sin(torch.matmul(x, self.W_x))], dim=1)
        H_t = torch.cat([torch.cos(torch.matmul(t, self.W_t)), 
                         torch.sin(torch.matmul(t, self.W_t))], dim=1)
        
        H_x = self.model_(H_x)
        H_t = self.model_(H_t)

        H = torch.multiply(H_x, H_t)
        u = self.last_layer_(H)
        return u


class STMFF1DT(nn.Module):
    r"""Spatio Temporal Multiscale Fourier Feature Model for 1D Space and Time.

    Args:
        nn_layers: The layer structure of the neural network.
            For example, [2, 100, 100, 100, 1].
            Note that nn_layers[1] must be an even number.
        sigma_x: The parameter of Fourier features for the spatial domain.
            Generally, it takes values like 1, 20, 50, 100, 200, ...
            The larger sigma_x is, the higher the frequency of the learned spatial domain.
        sigma_t_1: The parameter of Fourier features for the temporal domain.
            Generally, it takes values like 1, ...
            The larger sigma_t_1 is, the higher the frequency of the learned temporal domain.
        sigma_t_2: The parameter of Fourier features for the temporal domain.
            Generally, it takes values like 10, ...
            The larger sigma_t_2 is, the higher the frequency of the learned temporal domain.

    Note::
        1. nn_layers[0] is the original [x, t] input.
        2. [x, t] is transformed into H_x, H_t_1, and H_t_2 through Fourier Embedding.
        3. Among them, W_x, W_t_1, and W_t_2 are the parameters of Fourier Embedding.
        4. Then H_x, H_t_1, and H_t_2 are successively fed into the same network (e.g., MLP).
        5. Finally, H_1 = H_x * H_t_1 and H_2 = H_x * H_t_2 are used as the fused features.
        6. Concatenate H = [H_1, H_2] and pass it to the last linear layer to return solution.

    Example::
        >>> NN_LAYERS = [2, 100, 100, 100, 1]
        >>> network = STMFF1DT(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh',
            init_type: str = 'xavier_normal',
            sigma_x: int = 1,
            sigma_t_1: int = 1, 
            sigma_t_2: int = 10
    ):
        r"""Initialize the STMFF1DT model.

        Args:
            nn_layers (List[int]): The layer structure of the neural network, e.g., [2, 100, 100, 100, 1]. 
                Note that nn_layers[1] must be an even number.
            act_type (str, optional): The type of activation function. Defaults to 'tanh'.
            init_type (str, optional): The method for initializing network parameters. Defaults to 'xavier_normal'.
            sigma_x (int, optional): The parameter of Fourier features for the spatial domain. Defaults to 1.
            sigma_t_1 (int, optional): The first scale parameter of Fourier features for the temporal domain. Defaults to 1.
            sigma_t_2 (int, optional): The second scale parameter of Fourier features for the temporal domain. Defaults to 10.

        Attributes:
            args (dict): A dictionary storing all input arguments for reference.
            sigma_x (int): The parameter of Fourier features for the spatial domain.
            sigma_t_1 (int): The first scale parameter of Fourier features for the temporal domain.
            sigma_t_2 (int): The second scale parameter of Fourier features for the temporal domain.
            W_x (nn.Parameter): Fourier feature matrix for the spatial domain, non - trainable.
            W_t_1 (nn.Parameter): Fourier feature matrix for the first scale of the temporal domain, non - trainable.
            W_t_2 (nn.Parameter): Fourier feature matrix for the second scale of the temporal domain, non - trainable.
            model_ (nn.Sequential): A sequential container representing the neural network model excluding the last layer.
            last_layer_ (nn.Linear): The final linear layer of the network.
        """
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
            'sigma_x': sigma_x,
            'sigma_t_1': sigma_t_1,
            'sigma_t_2': sigma_t_2,
        }
        super(STMFF1DT, self).__init__()

        self.sigma_x = sigma_x
        self.sigma_t_1 = sigma_t_1
        self.sigma_t_2 = sigma_t_2
        self.W_x = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_t_1 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_t_1, requires_grad=False)
        self.W_t_2 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_t_2, requires_grad=False)
        
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(2 * nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, t = X[:, [0]], X[:, [1]]

        H_x = torch.cat([torch.cos(torch.matmul(x, self.W_x)), 
                         torch.sin(torch.matmul(x, self.W_x))], dim=1)
        H_t_1 = torch.cat([torch.cos(torch.matmul(t, self.W_t_1)), 
                           torch.sin(torch.matmul(t, self.W_t_1))], dim=1)
        H_t_2 = torch.cat([torch.cos(torch.matmul(t, self.W_t_2)), 
                           torch.sin(torch.matmul(t, self.W_t_2))], dim=1)
        
        H_x = self.model_(H_x)
        H_t_1 = self.model_(H_t_1)
        H_t_2 = self.model_(H_t_2)

        H_1 = torch.multiply(H_x, H_t_1)
        H_2 = torch.multiply(H_x, H_t_2)
        H = torch.cat([H_1, H_2], dim=1)
        u = self.last_layer_(H)
        return u


class FF2D(nn.Module):
    r"""Fourier Feature Model for 2D Space.

    Args:
        nn_layers: The layer structure of the neural network.
            For example, [2, 100, 100, 100, 1].
            Note that nn_layers[1] must be an even number.
        sigma_x: The parameter of Fourier features for the spatial domain.
            Generally, it takes values like 1, 20, 50, 100, 200, ...
            The larger sigma_x is, the higher the frequency of the learned spatial domain.
        sigma_y: The parameter of Fourier features for the spatial domain.
            Generally, it takes values like 1, 20, 50, 100, 200, ...
            The larger sigma_y is, the higher the frequency of the learned spatial domain.

    Note::
        1. nn_layers[0] is the original [x, y] input.
        2. [x, y] is transformed into H_x and H_y through Fourier Embedding.
        3. Among them, W_x and W_y are the parameters of Fourier Embedding.
        4. Then H_x and H_y are successively fed into the same network (e.g., MLP).
        5. Finally, H = H_x * H_y is passed as the fused feature to the last linear layer to return solution.

    Example::
        >>> NN_LAYERS = [2, 100, 100, 100, 1]
        >>> network = FF2D(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'xavier_normal',
            sigma_x: int = 10, 
            sigma_y: int = 10
    ):
        r"""Initialize the FF2D model.

        Args:
            nn_layers (List[int]): The layer structure of the neural network, e.g., [2, 100, 100, 100, 1]. 
                Note that nn_layers[1] must be an even number.
            act_type (str, optional): The type of activation function. Defaults to 'tanh'.
            init_type (str, optional): The method for initializing network parameters. Defaults to 'xavier_normal'.
            sigma_x (int, optional): The parameter of Fourier features for the spatial domain. Defaults to 10.
            sigma_y (int, optional): The parameter of Fourier features for the spatial domain. Defaults to 10.

        Attributes:
            args (dict): A dictionary storing all input arguments for reference.
            sigma_x (int): The parameter of Fourier features for the spatial domain.
            sigma_y (int): The parameter of Fourier features for the spatial domain.
            W_x (nn.Parameter): Fourier feature matrix for the spatial domain, non - trainable.
            W_y (nn.Parameter): Fourier feature matrix for the spatial domain, non - trainable.
            model_ (nn.Sequential): A sequential container representing the neural network model excluding the last layer.
            last_layer_ (nn.Linear): The final linear layer of the network.
        """
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
        }
        super(FF2D, self).__init__()

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.W_x = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_y = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_y, requires_grad=False)
        
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, y = X[:, [0]], X[:, [1]]

        H_x = torch.cat([torch.cos(torch.matmul(x, self.W_x)), 
                         torch.sin(torch.matmul(x, self.W_x))], dim=1)
        H_y = torch.cat([torch.cos(torch.matmul(y, self.W_y)), 
                         torch.sin(torch.matmul(y, self.W_y))], dim=1)
        
        H_x = self.model_(H_x)
        H_y = self.model_(H_y)

        H = torch.multiply(H_x, H_y)
        u = self.last_layer_(H)
        return u


class MFF2D(nn.Module):
    r"""Multiscale Fourier Feature Model for 2D Space.

    Args:
        nn_layers: The layer structure of the neural network.
            For example, [2, 100, 100, 100, 1].
            Note that nn_layers[1] must be an even number.
        sigma_x_1: The first parameter of Fourier features for the spatial domain in the x - direction.
            Generally, it takes values like 1, 20, ...
            The larger sigma_x is, the higher the frequency of the learned spatial domain.
        sigma_x_2: The second parameter of Fourier features for the spatial domain in the x - direction.
            Generally, it takes values like 10, 100,...
            The larger sigma_x is, the higher the frequency of the learned spatial domain.
        sigma_y_1: The first parameter of Fourier features for the spatial domain in the y - direction.
            Generally, it takes values like 1, 20, ...
            The larger sigma_y is, the higher the frequency of the learned spatial domain.
        sigma_y_2: The second parameter of Fourier features for the spatial domain in the y - direction.
            Generally, it takes values like 10, 100,...
            The larger sigma_y is, the higher the frequency of the learned spatial domain.

    Note::
        1. nn_layers[0] is the original [x, y] input.
        2. [x, y] is transformed into H_x_1, H_x_2, H_y_1, and H_y_2 through Fourier Embedding.
        3. Among them, W_x_1, W_x_2, W_y_1, and W_y_2 are the parameters of Fourier Embedding.
        4. Then H_x_1, H_x_2, H_y_1, and H_y_2 are successively fed into the same network (e.g., MLP).
        5. Finally, H = (H_x_1 * H_x_2) * (H_y_1 * H_y_2) is passed to the last linear layer to return solution.

    Example::
        >>> NN_LAYERS = [2, 100, 100, 100, 1]
        >>> network = MFF2D(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'xavier_normal',
            sigma_x_1: int = 1, 
            sigma_x_2: int = 10, 
            sigma_y_1: int = 1, 
            sigma_y_2: int = 10
    ):
        r"""Initialize the MFF2D model.

        Args:
            nn_layers (List[int]): The layer structure of the neural network. For example, [2, 100, 100, 100, 1]. 
                Note that nn_layers[1] must be an even number.
            act_type (str, optional): The type of activation function. Defaults to 'tanh'.
            init_type (str, optional): The method for initializing network parameters. Defaults to 'xavier_normal'.
            sigma_x_1 (int, optional): The first scale parameter of Fourier features for the x spatial domain. Defaults to 1.
            sigma_x_2 (int, optional): The second scale parameter of Fourier features for the x spatial domain. Defaults to 10.
            sigma_y_1 (int, optional): The first scale parameter of Fourier features for the y spatial domain. Defaults to 1.
            sigma_y_2 (int, optional): The second scale parameter of Fourier features for the y spatial domain. Defaults to 10.

        Attributes:
            args (dict): A dictionary storing all input arguments for reference.
            sigma_x_1 (int): The first scale parameter of Fourier features for the x spatial domain.
            sigma_x_2 (int): The second scale parameter of Fourier features for the x spatial domain.
            sigma_y_1 (int): The first scale parameter of Fourier features for the y spatial domain.
            sigma_y_2 (int): The second scale parameter of Fourier features for the y spatial domain.
            W_x_1 (nn.Parameter): Fourier feature matrix for the first scale of the x spatial domain, non - trainable.
            W_x_2 (nn.Parameter): Fourier feature matrix for the second scale of the x spatial domain, non - trainable.
            W_y_1 (nn.Parameter): Fourier feature matrix for the first scale of the y spatial domain, non - trainable.
            W_y_2 (nn.Parameter): Fourier feature matrix for the second scale of the y spatial domain, non - trainable.
            model_ (nn.Sequential): A sequential container representing the neural network model excluding the last layer.
            last_layer_ (nn.Linear): The final linear layer of the network.
        """
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
            'sigma_x_1': sigma_x_1,
            'sigma_x_2': sigma_x_2,
            'sigma_y_1': sigma_y_1,
            'sigma_y_2': sigma_y_2,
        }
        super(MFF2D, self).__init__()

        self.sigma_x_1 = sigma_x_1
        self.sigma_x_2 = sigma_x_2
        self.sigma_y_1 = sigma_y_1
        self.sigma_y_2 = sigma_y_2
        self.W_x_1 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x_1, requires_grad=False)
        self.W_x_2 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x_2, requires_grad=False)
        self.W_y_1 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_y_1, requires_grad=False)
        self.W_y_2 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_y_2, requires_grad=False)
        
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, y = X[:, [0]], X[:, [1]]

        H_x_1 = torch.cat([torch.cos(torch.matmul(x, self.W_x_1)), 
                           torch.sin(torch.matmul(x, self.W_x_1))], dim=1)
        H_x_2 = torch.cat([torch.cos(torch.matmul(x, self.W_x_2)), 
                           torch.sin(torch.matmul(x, self.W_x_2))], dim=1)
        H_y_1 = torch.cat([torch.cos(torch.matmul(y, self.W_y_1)), 
                           torch.sin(torch.matmul(y, self.W_y_1))], dim=1)
        H_y_2 = torch.cat([torch.cos(torch.matmul(y, self.W_y_2)), 
                           torch.sin(torch.matmul(y, self.W_y_2))], dim=1)
        
        H_x_1 = self.model_(H_x_1)
        H_x_2 = self.model_(H_x_2)
        H_y_1 = self.model_(H_y_1)
        H_y_2 = self.model_(H_y_2)

        H_x = torch.multiply(H_x_1, H_x_2)
        H_y = torch.multiply(H_y_1, H_y_2)
        H = torch.multiply(H_x, H_y)
        u = self.last_layer_(H)
        return u


class FF2DT(nn.Module):
    r"""Fourier Feature Model for 2D Space and Time.

    Args:
        nn_layers: The layer structure of the neural network.
            For example, [3, 100, 100, 100, 1].
            Note that nn_layers[1] must be an even number.
        sigma_x: The parameter of Fourier features for the spatial domain.
            Generally, it takes values like 1, 20, 50, 100, 200, ...
            The larger sigma_x is, the higher the frequency of the learned spatial domain.
        sigma_y: The parameter of Fourier features for the spatial domain.
            Generally, it takes values like 1, 20, 50, 100, 200, ...
            The larger sigma_y is, the higher the frequency of the learned spatial domain.
        sigma_t: The parameter of Fourier features for the temporal domain.
            Generally, it takes values like 1, 5, 10, ...
            The larger sigma_t is, the higher the frequency of the learned temporal domain.

    Note::
        1. nn_layers[0] is the original [x, y, t] input.
        2. [x, y, t] is transformed into H_x, H_y, and H_t through Fourier Embedding.
        3. Among them, W_x, W_y, and W_t are the parameters of Fourier Embedding.
        4. Then H_x, H_y, and H_t are successively fed into the same network (e.g., MLP).
        5. Finally, H = H_x * H_y * H_t is passed as the fused feature to the last linear layer to return u.

    Example::
        >>> NN_LAYERS = [3, 100, 100, 100, 1]
        >>> network = FF2DT(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'xavier_normal',
            sigma_x: int = 10, 
            sigma_y: int = 10, 
            sigma_t: int = 10
    ):
        r"""Initialize the FF2DT model.

        Args:
            nn_layers (List[int]): The layer structure of the neural network. For example, [3, 100, 100, 100, 1]. 
                Note that the second element of this list must be an even number.
            act_type (str, optional): The type of activation function to be used in the neural network. Defaults to 'tanh'. Other available options include 'ReLU', 'LeakyReLU', 'Sigmoid', etc.
            init_type (str, optional): The method for initializing the network parameters. Defaults to 'xavier_normal'. Other options are 'kaiming_normal', 'xavier_uniform', etc.
            sigma_x (int, optional): The parameter of Fourier features for the spatial domain in the x - direction. Larger values lead to a higher frequency of the learned spatial domain. Defaults to 10.
            sigma_y (int, optional): The parameter of Fourier features for the spatial domain in the y - direction. Larger values lead to a higher frequency of the learned spatial domain. Defaults to 10.
            sigma_t (int, optional): The parameter of Fourier features for the temporal domain. Larger values lead to a higher frequency of the learned temporal domain. Defaults to 10.

        Attributes:
            args (dict): A dictionary that stores all input arguments for reference.
            sigma_x (int): The parameter of Fourier features for the spatial domain in the x - direction.
            sigma_y (int): The parameter of Fourier features for the spatial domain in the y - direction.
            sigma_t (int): The parameter of Fourier features for the temporal domain.
            W_x (nn.Parameter): Fourier feature matrix for the spatial domain in the x - direction, which is non - trainable.
            W_y (nn.Parameter): Fourier feature matrix for the spatial domain in the y - direction, which is non - trainable.
            W_t (nn.Parameter): Fourier feature matrix for the temporal domain, which is non - trainable.
            model_ (nn.Sequential): A sequential container representing the neural network model excluding the last layer.
            last_layer_ (nn.Linear): The final linear layer of the network.
        """
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'sigma_t': sigma_t,
        }
        super(FF2D, self).__init__()

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_t = sigma_t
        self.W_x = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_y = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_y, requires_grad=False)
        self.W_t = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_t, requires_grad=False)

        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, y, t = X[:, [0]], X[:, [1]], X[:, [-1]]

        H_x = torch.cat([torch.cos(torch.matmul(x, self.W_x)), 
                         torch.sin(torch.matmul(x, self.W_x))], dim=1)
        H_y = torch.cat([torch.cos(torch.matmul(y, self.W_y)), 
                         torch.sin(torch.matmul(y, self.W_y))], dim=1)
        H_t = torch.cat([torch.cos(torch.matmul(t, self.W_t)),
                         torch.sin(torch.matmul(t, self.W_t))], dim=1)
        
        H_x = self.model_(H_x)
        H_y = self.model_(H_y)
        H_t = self.model_(H_t)

        H = torch.multiply(H_x, H_y)
        H = torch.multiply(H, H_t)
        u = self.last_layer_(H)
        return u
