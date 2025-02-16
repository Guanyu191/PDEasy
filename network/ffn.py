r"""Fourier Feature Network 类神经网络模型.

Fourier Feature 类神经网络模型参考自
https://www.sciencedirect.com/science/article/pii/S0045782521002759
包括 3 类模型：
    1. Multiscale Fourier Feature Model (MFF).
    2. Spatio Temporal Fourier Feature Model (STFF).
    3. Spatio Temporal Multiscale Fourier Feature Model (STMFF).

具体地, 我们实现了针对 1D, 2D, 1DT, 2DT 的问题的 FFN 模型, 包括:
    1. MFF1D: Multiscale Fourier Feature Model for 1D Space.
    2. STFF1DT: Spatio Temporal Fourier Feature Model for 1D Space and Time.
    3. STMFF1DT: Spatio Temporal Multiscale Fourier Feature Model for 1D Space and Time.
    4. FF2D: Fourier Feature Model for 2D Space.
    5. MFF2D: Multiscale Fourier Feature Model for 2D Space.
    6. FF2DT: Fourier Feature Model for 2D Space and Time.

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
    >>> NN_LAYERS = [1, 100, 100, 1]
    >>> network = MFF1D(NN_LAYERS)
    >>> ...
"""

r'''第一版注释
Descripttion: 
    Replicating the code from the paper:
        On the eigenvector bias of Fourier feature networks: From regression 
            to solving multi-scale PDEs with physics-informed neural networks
        https://www.sciencedirect.com/science/article/pii/S0045782521002759
    Here are three Fourier feature network for solving high frequencies problem.
        1. Multiscale Fourier Feature Model
        2. Spatio Temporal Fourier Feature Model
        3. Spatio Temporal Multiscale Fourier Feature Model
Author: Guanyu
Date: 2025-02-04 17:04:02
LastEditTime: 2025-02-04 17:15:23
FilePath: \FourierFeaturePINN\fourier_feature_network.py
'''
import torch
import torch.nn as nn

from utils.init_network_weights import init_network_weights
from utils.init_network_activation_function import init_network_activation_function

from typing import List, Tuple, Union
from torch import Tensor


class MFF1D(nn.Module):
    r"""Multiscale Fourier Feature Model for 1D Space.

    Args:
        nn_layers (List[int]): 神经网络的层结构
            例如 [1, 100, 100, 100, 1]
            但是 nn_layers[1] 必须是偶数
            注意到
        sigma_1 (int, optional): 傅里叶特征的参数
            第一个尺度一般取 1, ...
            sigma_1 越大，学习的时空域整体频率越高
        sigma_2 (int, optional): 傅里叶特征的参数
            第二个尺度一般取 10, 20, 50, 100, 200, ...
            sigma_2 越大，学习的时空域整体频率越高

    Note::
        1. nn_layers[0] 是原始的 [x] 输入
        2. [x] 经过 Fourier Embedding 变为 H_1 和 H_2
        3. 其中 W_1 和 W_2 是 Fourier Embedding 的参数
        4. 进而 H_1 和 H_2 先后传入同一个网络 (例如 MLP)
        5. 最后将 H = [H_1, H_2] 作为拼接特征传给最后的线性层返回 u

    Example::
        >>> NN_LAYERS = [1, 100, 100, 100, 1]
        >>> network = MFF1D(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh',
            init_type: str = 'defalut',
            sigma_1: int = 1, 
            sigma_2: int = 10,
    ) -> None:
        r"""_summary_

        Args:
            nn_layers (List[int]): _description_
            act_type (str, optional): _description_. Defaults to 'tanh'.
            init_type (str, optional): _description_. Defaults to 'defalut'.
            sigma_1 (int, optional): _description_. Defaults to 1.
            sigma_2 (int, optional): _description_. Defaults to 10.
        """

        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
            'sigma_1': sigma_1,
            'sigma_2': sigma_2
        }
        super(MFF1D, self).__init__()

        # 傅里叶特征初始化
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.W_1 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_1, requires_grad=False)
        self.W_2 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_2, requires_grad=False)
        
        # 网络初始化
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i+1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i+1}', init_network_activation_function(act_type))

        self.last_layer_ = nn.Linear(2 * nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        # 傅里叶特征编码
        H_1 = torch.cat([torch.cos(torch.matmul(X, self.W_1)), 
                         torch.sin(torch.matmul(X, self.W_1))], dim=1)
        H_2 = torch.cat([torch.cos(torch.matmul(X, self.W_2)), 
                         torch.sin(torch.matmul(X, self.W_2))], dim=1)
        
        # 空间和时间特征分别 Passing through
        H_1 = self.model_(H_1)
        H_2 = self.model_(H_2)

        # 拼接多尺度特征
        H = torch.cat([H_1, H_2], dim=1)
        u = self.last_layer_(H)
        return u


class STFF1DT(nn.Module):
    r"""Spatio Temporal Fourier Feature Model for 1D Space and Time.

    Args:
        nn_layers: 神经网络的层结构
            例如 [2, 100, 100, 100, 1]
            但是 nn_layers[1] 必须是偶数
        sigma_x: 傅里叶特征的参数
            空间域一般取 1, 20, 50, 100, 200, ...
            sigma_x 越大，学习的空间域的频率越高
        sigma_t: 傅里叶特征的参数
            时间域一般取 1, ...
            sigma_t 越大，学习的时间域的频率越高

    Note::
        1. nn_layers[0] 是原始的 [x, t] 输入
        2. [x, t] 经过 Fourier Embedding 变为 H_x 和 H_t
        3. 其中 W_x 和 W_t 是 Fourier Embedding 的参数
        4. 进而 H_x 和 H_t 先后传入同一个网络 (例如 MLP)
        5. 最后将 H = H_x * H_t 作为融合特征传给最后的线性层返回 u

    Example::
        >>> NN_LAYERS = [2, 100, 100, 100, 1]
        >>> network = STFF1DT(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh',
            init_type: str = 'defalut',
            sigma_x: int = 1, 
            sigma_t: int = 10
    ) -> None:
        r"""_summary_

        Args:
            nn_layers (List[int]): _description_
            act_type (str, optional): _description_. Defaults to 'tanh'.
            init_type (str, optional): _description_. Defaults to 'defalut'.
            sigma_x (int, optional): _description_. Defaults to 1.
            sigma_t (int, optional): _description_. Defaults to 10.
        """

        super(STFF1DT, self).__init__()

        # 傅里叶特征初始化
        self.sigma_x = sigma_x
        self.sigma_t = sigma_t
        self.W_x = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_t = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_t, requires_grad=False)
        
        # 网络初始化
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, t = X[:, [0]], X[:, [1]]

        # 傅里叶特征编码
        H_x = torch.cat([torch.cos(torch.matmul(x, self.W_x)), 
                         torch.sin(torch.matmul(x, self.W_x))], dim=1)
        H_t = torch.cat([torch.cos(torch.matmul(t, self.W_t)), 
                         torch.sin(torch.matmul(t, self.W_t))], dim=1)
        
        # 空间和时间特征分别 Passing through
        H_x = self.model_(H_x)
        H_t = self.model_(H_t)

        # 融合时空特征
        # 类似于 f(x, t) = \sum_{k=-\infty}^{\infty} \hat{f}_k(t) e^{ikx}
        H = torch.multiply(H_x, H_t)
        u = self.last_layer_(H)
        return u


class STMFF1DT(nn.Module):
    r"""Spatio Temporal Multiscale Fourier Feature Model for 1D Space and Time.

    Args:
        nn_layers: 神经网络的层结构
            例如 [2, 100, 100, 100, 1]
            但是 nn_layers[1] 必须是偶数
        sigma_x: 傅里叶特征的参数
            空间域一般取 1, 20, 50, 100, 200, ...
            sigma_x 越大，学习的空间域的频率越高
        sigma_t_1: 傅里叶特征的参数
            时间域一般取 1, ...
            sigma_t_1 越大，学习的时间域的频率越高
        sigma_t_2: 傅里叶特征的参数
            时间域一般取 10, ...
            sigma_t_2 越大，学习的时间域的频率越高
    
    Note::
        1. nn_layers[0] 是原始的 [x, t] 输入
        2. [x, t] 经过 Fourier Embedding 变为 H_x 和 H_t_1, H_t_2
        3. 其中 W_x 和 W_t_1, W_t_2 是 Fourier Embedding 的参数
        4. 进而 H_x 和 H_t_1, H_t_2 先后传入同一个网络 (例如 MLP)
        5. 最后将 H_1 = H_x * H_t_1, H_2 = H_x * H_t_2 作为融合特征
        6. 拼接 H = [H_1, H_2] 传给最后的线性层返回 u

    Example::
        >>> NN_LAYERS = [2, 100, 100, 100, 1]
        >>> network = STMFF1DT(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh',
            init_type: str = 'defalut',
            sigma_x: int = 1,
            sigma_t_1: int = 1, 
            sigma_t_2: int = 10
    ) -> None:
        r"""_summary_

        Args:
            nn_layers (List[int]): _description_
            act_type (str, optional): _description_. Defaults to 'tanh'.
            init_type (str, optional): _description_. Defaults to 'defalut'.
            sigma_x (int, optional): _description_. Defaults to 1.
            sigma_t_1 (int, optional): _description_. Defaults to 1.
            sigma_t_2 (int, optional): _description_. Defaults to 10.
        """

        super(STMFF1DT, self).__init__()

        # 傅里叶特征初始化
        self.sigma_x = sigma_x
        self.sigma_t_1 = sigma_t_1
        self.sigma_t_2 = sigma_t_2
        self.W_x = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_t_1 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_t_1, requires_grad=False)
        self.W_t_2 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_t_2, requires_grad=False)
        
        # 网络初始化
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(2 * nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, t = X[:, [0]], X[:, [1]]

        # 傅里叶特征编码
        H_x = torch.cat([torch.cos(torch.matmul(x, self.W_x)), 
                         torch.sin(torch.matmul(x, self.W_x))], dim=1)
        H_t_1 = torch.cat([torch.cos(torch.matmul(t, self.W_t_1)), 
                           torch.sin(torch.matmul(t, self.W_t_1))], dim=1)
        H_t_2 = torch.cat([torch.cos(torch.matmul(t, self.W_t_2)), 
                           torch.sin(torch.matmul(t, self.W_t_2))], dim=1)
        
        # 空间和时间特征分别 Passing through
        H_x = self.model_(H_x)
        H_t_1 = self.model_(H_t_1)
        H_t_2 = self.model_(H_t_2)

        # 融合时空特征
        # 类似于 f(x, t) = \sum_{k=-\infty}^{\infty} \hat{f}_k(t) e^{ikx}
        H_1 = torch.multiply(H_x, H_t_1)
        H_2 = torch.multiply(H_x, H_t_2)
        H = torch.cat([H_1, H_2], dim=1)
        u = self.last_layer_(H)
        return u


class FF2D(nn.Module):
    r"""Fourier Feature Model for 2D Space.

    Args:
        nn_layers: 神经网络的层结构
            例如 [2, 100, 100, 100, 1]
            但是 nn_layers[1] 必须是偶数
        sigma_x: 傅里叶特征的参数
            空间域一般取 1, 20, 50, 100, 200, ...
            sigma_x 越大，学习的空间域的频率越高
        sigma_y: 傅里叶特征的参数
            空间域一般取 1, 20, 50, 100, 200...
            sigma_y 越大，学习的时间域的频率越高

    Note::
        1. nn_layers[0] 是原始的 [x, y] 输入
        2. [x, y] 经过 Fourier Embedding 变为 H_x 和 H_y
        3. 其中 W_x 和 W_y 是 Fourier Embedding 的参数
        4. 进而 H_x 和 H_y 先后传入同一个网络 (例如 MLP)
        5. 最后将 H = H_x * H_y 作为融合特征传给最后的线性层返回 u

    Example::
        >>> NN_LAYERS = [2, 100, 100, 100, 1]
        >>> network = FF2D(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'defalut',
            sigma_x: int = 10, 
            sigma_y: int = 10
        ) -> None:
        r"""_summary_

        Args:
            nn_layers (List[int]): _description_
            act_type (str, optional): _description_. Defaults to 'tanh'.
            init_type (str, optional): _description_. Defaults to 'defalut'.
            sigma_x (int, optional): _description_. Defaults to 10.
            sigma_y (int, optional): _description_. Defaults to 10.
        """

        super(FF2D, self).__init__()

        # 傅里叶特征初始化
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.W_x = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_y = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_y, requires_grad=False)
        
        # 网络初始化
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, y = X[:, [0]], X[:, [1]]

        # 傅里叶特征编码
        H_x = torch.cat([torch.cos(torch.matmul(x, self.W_x)), 
                         torch.sin(torch.matmul(x, self.W_x))], dim=1)
        H_y = torch.cat([torch.cos(torch.matmul(y, self.W_y)), 
                         torch.sin(torch.matmul(y, self.W_y))], dim=1)
        
        # 空间和时间特征分别 Passing through
        H_x = self.model_(H_x)
        H_y = self.model_(H_y)

        # 拼接傅里叶特征
        H = torch.multiply(H_x, H_y)
        u = self.last_layer_(H)
        return u


class MFF2D(nn.Module):
    r"""Multiscale Fourier Feature Model for 2D Space.

    Args:
        nn_layers: 神经网络的层结构
            例如 [2, 100, 100, 100, 1]
            但是 nn_layers[1] 必须是偶数
        sigma_x: 傅里叶特征的参数
            空间域一般取 1, 20, 50, 100, 200, ...
            sigma_x 越大，学习的空间域的频率越高
        sigma_t: 傅里叶特征的参数
            空间域一般取 1, ...
            sigma_t 越大，学习的时间域的频率越高

    Note::
        1. nn_layers[0] 是原始的 [x, y] 输入
        2. [x, y] 经过 Fourier Embedding 变为 H_x_1, H_x_2 和 H_y_1, H_y_2
        3. 其中 W_x_1, W_x_2 和 W_y_1, W_y_2 是 Fourier Embedding 的参数
        4. 进而 H_x_1, H_x_2 和 H_y_1, H_y_2 先后传入同一个网络 (例如 MLP)
        5. 最后将 H = [H_x_1, H_x_2, H_y_1, H_y_2] 作为拼接特征传给最后的线性层返回 u

    Example::
        >>> NN_LAYERS = [2, 100, 100, 100, 1]
        >>> network = MFF2D(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'default',
            sigma_x_1: int = 1, 
            sigma_x_2: int = 10, 
            sigma_y_1: int = 1, 
            sigma_y_2: int = 10
    ) -> None:
        r"""_summary_

        Args:
            nn_layers (List[int]): _description_
            act_type (str, optional): _description_. Defaults to 'tanh'.
            init_type (str, optional): _description_. Defaults to 'default'.
            sigma_x_1 (int, optional): _description_. Defaults to 1.
            sigma_x_2 (int, optional): _description_. Defaults to 10.
            sigma_y_1 (int, optional): _description_. Defaults to 1.
            sigma_y_2 (int, optional): _description_. Defaults to 10.
        """

        super(MFF2D, self).__init__()

        # 傅里叶特征初始化
        self.sigma_x_1 = sigma_x_1
        self.sigma_x_2 = sigma_x_2
        self.sigma_y_1 = sigma_y_1
        self.sigma_y_2 = sigma_y_2
        self.W_x_1 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x_1, requires_grad=False)
        self.W_x_2 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x_2, requires_grad=False)
        self.W_y_1 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_y_1, requires_grad=False)
        self.W_y_2 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_y_2, requires_grad=False)
        
        # 网络初始化
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, y = X[:, [0]], X[:, [1]]

        # 傅里叶特征编码
        H_x_1 = torch.cat([torch.cos(torch.matmul(x, self.W_x_1)), 
                           torch.sin(torch.matmul(x, self.W_x_1))], dim=1)
        H_x_2 = torch.cat([torch.cos(torch.matmul(x, self.W_x_2)), 
                           torch.sin(torch.matmul(x, self.W_x_2))], dim=1)
        H_y_1 = torch.cat([torch.cos(torch.matmul(y, self.W_y_1)), 
                           torch.sin(torch.matmul(y, self.W_y_1))], dim=1)
        H_y_2 = torch.cat([torch.cos(torch.matmul(y, self.W_y_2)), 
                           torch.sin(torch.matmul(y, self.W_y_2))], dim=1)
        
        # 空间和时间特征分别 Passing through
        H_x_1 = self.model_(H_x_1)
        H_x_2 = self.model_(H_x_2)
        H_y_1 = self.model_(H_y_1)
        H_y_2 = self.model_(H_y_2)

        # 拼接傅里叶特征
        H_x = torch.multiply(H_x_1, H_x_2)
        H_y = torch.multiply(H_y_1, H_y_2)
        H = torch.multiply(H_x, H_y)
        u = self.last_layer_(H)
        return u


class FF2DT(nn.Module):
    r"""Fourier Feature Model for 2D Space and Time.

    Args:
        nn_layers: 神经网络的层结构
            例如 [3, 100, 100, 100, 1]
            但是 nn_layers[1] 必须是偶数
        sigma_x: 傅里叶特征的参数
            空间域一般取 1, 20, 50, 100, 200, ...
            sigma_x 越大，学习的空间域的频率越高
        sigma_y: 傅里叶特征的参数
            空间域一般取 1, 20, 50, 100, 200...
            sigma_y 越大，学习的时间域的频率越高
        sigma_t: 傅里叶特征的参数
            时间域一般取 1, ...
            sigma_t 越大，学习的时间域的频率越高

    Note::
        1. nn_layers[0] 是原始的 [x, y, t] 输入
        2. [x, y, t] 经过 Fourier Embedding 变为 H_x, H_y, H_t
        3. 其中 W_x, W_y, W_t 是 Fourier Embedding 的参数
        4. 进而 H_x, H_y, H_t 先后传入同一个网络 (例如 MLP)
        5. 最后将 H = H_x * H_y * H_t 作为融合特征传给最后的线性层返回 u

    Example::
        >>> NN_LAYERS = [3, 100, 100, 100, 1]
        >>> network = FF2D(NN_LAYERS)
    """
    def __init__(
            self, 
            nn_layers: List[int], 
            act_type: str = 'tanh', 
            init_type: str = 'default',
            sigma_x: int = 10, 
            sigma_y: int = 10, 
            sigma_t: int = 10
    ) -> None:
        r"""_summary_

        Args:
            nn_layers (List[int]): _description_
            act_type (str, optional): _description_. Defaults to 'tanh'.
            init_type (str, optional): _description_. Defaults to 'default'.
            sigma_x (int, optional): _description_. Defaults to 10.
            sigma_y (int, optional): _description_. Defaults to 10.
            sigma_t (int, optional): _description_. Defaults to 10.
        """

        super(FF2D, self).__init__()

        # 傅里叶特征初始化
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_t = sigma_t
        self.W_x = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_y = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_y, requires_grad=False)
        self.W_t = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_t, requires_grad=False)

        # 网络初始化
        self.model_ = nn.Sequential()
        nn_layers_ = nn_layers[1:-1]
        for i in range(len(nn_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(nn_layers_[i], nn_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', init_network_activation_function(act_type))
        self.last_layer_ = nn.Linear(nn_layers_[-1], nn_layers[-1], bias=False)

        self.apply(lambda module: init_network_weights(module, init_type))

    def forward(self, X):
        x, y, t = X[:, [0]], X[:, [1]], X[:, [-1]]

        # 傅里叶特征编码
        H_x = torch.cat([torch.cos(torch.matmul(x, self.W_x)), 
                         torch.sin(torch.matmul(x, self.W_x))], dim=1)
        H_y = torch.cat([torch.cos(torch.matmul(y, self.W_y)), 
                         torch.sin(torch.matmul(y, self.W_y))], dim=1)
        H_t = torch.cat([torch.cos(torch.matmul(t, self.W_t)),
                         torch.sin(torch.matmul(t, self.W_t))], dim=1)
        
        # 空间和时间特征分别 Passing through
        H_x = self.model_(H_x)
        H_y = self.model_(H_y)
        H_t = self.model_(H_t)

        # 拼接傅里叶特征
        H = torch.multiply(H_x, H_y)
        H = torch.multiply(H, H_t)
        u = self.last_layer_(H)
        return u
