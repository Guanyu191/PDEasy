r"""Multi-Head Network 类神经网络模型.

Multi-Head Network 类神经网络模型包括:
    1. 最后两层分头输出的 MHN.
    2. TODO

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
    >>> NN_LAYERS = [2, 20, 20, 3]
    >>> network = MHN(NN_LAYERS)
"""
import torch
import torch.nn as nn

from utils.init_network_weights import init_network_weights
from utils.init_network_activation_function import init_network_activation_function

from typing import List, Tuple, Union
from torch import Tensor


class MHN(nn.Module):
    """Multi-Head Network.

    在输出时采用多头分别输出, 适合多输出多尺度问题.

    注意, len(nn_layers) >= 4

    Example::
        >>> NN_LAYERS = [2, 20, 20, 3]
        >>> network = MHN(nn_layers)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'defalut',
        ) -> None:
        """通过 list 初始化神经网络.

        Args:
            nn_layers (List[int]): 表示神经网络层结构的 list.
            act_type (str, optional): 激活函数. Defaults to 'tanh'.
            init_type (str, optional): 网络参数初始化方法. Defaults to 'defalut'.
        """

        super(MHN, self).__init__()
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
        }

        self.model = nn.Sequential()
        for i in range(len(nn_layers)-3):  # len(nn_layers) >= 4
            layer = nn.Sequential()
            layer.add_module(f'fc{i+1}', nn.Linear(nn_layers[i], nn_layers[i+1], bias=True))
            layer.add_module(f'act{i+1}', init_network_activation_function(act_type))
            self.model.add_module(f'layer{i+1}', layer)

        # 利用 ModuleList 构造多头
        self.heads = nn.ModuleList()
        n_outputs = nn_layers[-1]
        n_hiddens = nn_layers[-2] // n_outputs
        for i in range(nn_layers[-1]):
            h = nn.Sequential()
            layer = nn.Sequential()
            layer.add_module(f'fc_head_hidden', nn.Linear(nn_layers[-3], n_hiddens, bias=True))
            layer.add_module(f'act_head_hidden', init_network_activation_function(act_type))
            h.add_module(f'layer_head_hidden', layer)
            h.add_module(f'head{i+1}', nn.Linear(n_hiddens, 1, bias=False))
            self.heads.append(h)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        X = self.model(X)

        # 分别经过各个头 拼接输出
        out = []
        for h in self.heads:
            out.append(h(X))
        out = torch.cat(out, dim=1)
        return out
