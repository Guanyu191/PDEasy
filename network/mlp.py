r"""MLP type neural network models.

The MLP type neural network models include:
    1. Ordinary MLP.
    2. Improved ModifiedMLP, referenced from https://epubs.siam.org/doi/10.1137/20M1318043

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
    >>> NN_LAYERS = [2, 20, 20, 1]
    >>> network = MLP(NN_LAYERS)
    >>> ...
    >>> network = ModifiedMLP(NN_LAYERS)
"""
import torch
import torch.nn as nn

from utils.init_network_weights import init_network_weights
from utils.init_network_activation_function import init_network_activation_function

from typing import List, Tuple, Union
from torch import Tensor


class MLP(nn.Module):
    r"""Muti-layer perceptron (MLP) / fully-connected neural network (FNN).

    The most fundamental neural network model.

    Example::
        >>> NN_LAYERS = [2, 20, 20, 1]
        >>> network = MLP(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'xavier_normal',
    ):
        r"""
        Initialize a Multi - layer Perceptron (MLP) neural network.

        Args:
            nn_layers (List[int]): A list representing the layer structure of the neural network. 
                For example, [2, 20, 20, 1] means 2 neurons in the input layer, 
                two hidden layers with 20 neurons each, and 1 neuron in the output layer.
            act_type (str, optional): The type of activation function. Defaults to 'tanh'. 
                Other available options include 'ReLU', 'LeakyReLU', etc.
            init_type (str, optional): The method for initializing network parameters. 
                Defaults to 'xavier_normal'. Other options include 'kaiming_normal', etc.

        Attributes:
            args (dict): A dictionary storing all input arguments for later reference.
            model (nn.Sequential): A sequential container that holds the entire neural network model, including all layers.
        """

        super(MLP, self).__init__()
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
        }

        self.model = nn.Sequential()
        for i in range(len(nn_layers) - 2):
            layer = nn.Sequential()
            layer.add_module(f'fc_{i+1}', nn.Linear(nn_layers[i], nn_layers[i+1], bias=True))
            layer.add_module(f'act_{i+1}', init_network_activation_function(act_type))
            self.model.add_module(f'layer_{i+1}', layer)

        layer = nn.Sequential()
        layer.add_module(f'fc_{len(nn_layers)-1}', nn.Linear(nn_layers[-2], nn_layers[-1], bias=False))
        self.model.add_module(f'layer_{len(nn_layers)-1}', layer)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)


class ModifiedMLP(nn.Module):
    r"""Modified Multi-layer perceptron (MMLP).

    An attention mechanism is added to the MLP.
    Referenced from: https://epubs.siam.org/doi/10.1137/20M1318043

    Example::
        >>> NN_LAYERS = [2, 20, 20, 1]
        >>> network = ModifiedMLP(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh',
            init_type: str = 'xavier_normal',
    ):
        r"""
        Initialize a Modified Multi - layer Perceptron (ModifiedMLP) neural network.

        Args:
            nn_layers (List[int]): A list representing the layer structure of the neural network. 
                For example, [2, 20, 20, 1] indicates 2 neurons in the input layer, 
                two hidden layers with 20 neurons each, and 1 neuron in the output layer.
            act_type (str, optional): The type of activation function to be used in the network. 
                Defaults to 'tanh'. Other available options are 'ReLU', 'LeakyReLU', etc.
            init_type (str, optional): The method for initializing the network parameters. 
                Defaults to 'xavier_normal'. Other options include 'kaiming_normal', etc.

        Attributes:
            args (dict): A dictionary that stores all the input arguments for future reference.
            encoder_u (nn.Sequential): A sequential container for the encoder 'u' that processes the input data.
            encoder_v (nn.Sequential): A sequential container for the encoder 'v' that processes the input data.
            model (nn.Sequential): A sequential container that holds the main part of the neural network model, excluding the encoders.
        """

        super(ModifiedMLP, self).__init__()
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
        }

        self.encoder_u = nn.Sequential()
        self.encoder_u.add_module('fc_u', nn.Linear(nn_layers[0], nn_layers[1], bias=True))
        self.encoder_u.add_module('act_u', init_network_activation_function(act_type))

        self.encoder_v = nn.Sequential()
        self.encoder_v.add_module('fc_v', nn.Linear(nn_layers[0], nn_layers[1], bias=True))
        self.encoder_v.add_module('act_v', init_network_activation_function(act_type))

        self.model = nn.Sequential()
        for i in range(len(nn_layers)-2):
            layer = nn.Sequential()
            layer.add_module(f'fc_{i+1}', nn.Linear(nn_layers[i], nn_layers[i+1], bias=True))
            layer.add_module(f'act_{i+1}', init_network_activation_function(act_type))
            self.model.add_module(f'layer_{i+1}', layer)

        last_layer = nn.Sequential()
        last_layer.add_module(
            f'fc_{len(nn_layers)-1}', nn.Linear(nn_layers[-2], nn_layers[-1], bias=False))
        self.model.add_module(f'layer_{len(nn_layers)-1}', last_layer)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X: Tensor) -> Tensor:
        u = self.encoder_u(X)
        v = self.encoder_v(X)

        for i in range(len(self.model) - 1):
            X = self.model[i](X)
            X = X / 2.
            X = (1 - X) * u + X * v
        return self.model[-1](X)
