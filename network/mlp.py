r"""MLP 类神经网络模型.

MLP 类神经网络模型包括:
    1. 普通的 MLP.
    2. 改进的 ModifiedMLP, 参考自 https://epubs.siam.org/doi/10.1137/20M1318043

生成网络实例仅需要传入一个 list, 表示神经网络的层数和每层的神经元数量.
另外, 可以调整激网络的激活函数, 以及网络参数初始化的方法.

网络的激活函数默认为 Tanh, 其余可以选择:
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

网络参数初始化方法默认为 default, 即采用 PyTorch 自带的初始化方案, 其余可以选择:
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
    >>> mlp = MLP(NN_LAYERS)
    >>> mlp = ModifiedMLP(NN_LAYERS)
"""
import torch
import torch.nn as nn

from utils.init_network_weights import init_network_weights
from utils.init_network_activation_function import init_network_activation_function

from typing import List, Tuple, Union
from torch import Tensor


class MLP(nn.Module):
    r"""Muti-layer perceptron (MLP) / fully-connected neural network (FNN).

    最基础的神经网络模型.

    Example::
        >>> NN_LAYERS = [2, 20, 20, 1]
        >>> mlp = MLP(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'defalut',
    ) -> None:
        r"""通过 list 初始化神经网络.

        Args:
            nn_layers (List[int]): 表示神经网络层结构的 list.
            act_type (str, optional): 激活函数. Defaults to 'tanh'.
            init_type (str, optional): 网络参数初始化方法. Defaults to 'defalut'.
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

    对 MLP 增加了注意力机制.
    参考自: https://epubs.siam.org/doi/10.1137/20M1318043

    Example::
        >>> NN_LAYERS = [2, 20, 20, 1]
        >>> mlp = ModifiedMLP(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh',
            init_type: str = 'defalut',
    ) -> None:
        r"""通过 list 初始化神经网络.

        Args:
            nn_layers (List[int]): 表示神经网络层结构的 list.
            act_type (str, optional): 激活函数. Defaults to 'tanh'.
            init_type (str, optional): 网络参数初始化方法. Defaults to 'defalut'.
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
