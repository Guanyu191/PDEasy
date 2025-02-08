r'''
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


class MFF(nn.Module):
    r"""
    Multiscale Fourier Feature Model
    模型说明：
        nn_layers[0] 是原始的 [x, t] 输入
        [x, t] 经过 Fourier Embedding 变为 H_1 和 H_2
        其中 W_1 和 W_2 是 Fourier Embedding 的参数
        进而 H_1 和 H_2 先后传入同一个网络 (例如 MLP)
        最后将 H = [H_1, H_2] 作为拼接特征传给最后的线性层返回 u
    ---
    参数说明：
        nn_layers: 神经网络的层结构
            例如 [2, 100, 100, 100, 1]
            但是 nn_layers[1] 必须是偶数
        sigma_1: 傅里叶特征的参数
            第一个尺度一般取 1, ...
            sigma_1 越大，学习的时空域整体频率越高
        sigma_2: 傅里叶特征的参数
            第二个尺度一般取 10, 20, 50, 100, 200, ...
            sigma_2 越大，学习的时空域整体频率越高
    ---
    示例：
        NN_LAYERS = [2, 100, 100, 100, 1]
        network = MFF(NN_LAYERS)
    """
    def __init__(self, nn_layers, act_type='tanh', init_type='default',
                 sigma_1=1, sigma_2=10):
        self.args = {
            'nn_layers': nn_layers,
            'act_type': act_type,
            'init_type': init_type,
            'sigma_1': sigma_1,
            'sigma_2': sigma_2
        }
        super(MFF, self).__init__()

        # 傅里叶特征初始化
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.W_1 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_1, requires_grad=False)
        self.W_2 = nn.Parameter(torch.randn(1, nn_layers[1] // 2) * sigma_2, requires_grad=False)
        
        # 网络初始化
        self.model_ = nn.Sequential()
        mlp_layers_ = nn_layers[1:-1]
        for i in range(len(mlp_layers_) - 1):
            self.model_.add_module(f'fc{i+1}', nn.Linear(mlp_layers_[i], mlp_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i+1}', init_network_activation_function(act_type))

        self.last_layer_ = nn.Linear(2 * mlp_layers_[-1], nn_layers[-1], bias=False)

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


class STFF(nn.Module):
    def __init__(self, mlp_layers, sigma_x=200, sigma_t=1):
        """
        Spatio Temporal Fourier Feature Model
        模型说明：
            mlp_layers[0] 是原始的 [x, t] 输入
            [x, t] 经过 Fourier Embedding 变为 H_x 和 H_t
            其中 W_x 和 W_t 是 Fourier Embedding 的参数
            进而 H_x 和 H_t 先后传入同一个网络 (例如 MLP)
            最后将 H = H_x * H_t 作为融合特征传给最后的线性层返回 u
        ---
        参数说明：
            mlp_layers: 神经网络的层结构
                例如 [2, 100, 100, 100, 1]
                但是 mlp_layers[1] 必须是偶数
            sigma_x: 傅里叶特征的参数
                空间域一般取 1, 20, 50, 100, 200, ...
                sigma_x 越大，学习的空间域的频率越高
            sigma_t: 傅里叶特征的参数
                时间域一般取 1, ...
                sigma_t 越大，学习的时间域的频率越高
        ---
        示例：
            NN_LAYERS = [2, 100, 100, 100, 1]
            network = STMFF(NN_LAYERS)
        """
        super(STFF, self).__init__()

        # 傅里叶特征初始化
        self.sigma_x = sigma_x
        self.sigma_t = sigma_t
        self.W_x = nn.Parameter(torch.randn(1, mlp_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_t = nn.Parameter(torch.randn(1, mlp_layers[1] // 2) * sigma_t, requires_grad=False)
        
        # 网络初始化
        self.model_ = nn.Sequential()
        mlp_layers_ = mlp_layers[1:-1]
        for i in range(len(mlp_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(mlp_layers_[i], mlp_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', nn.Tanh())
        self.last_layer_ = nn.Linear(mlp_layers_[-1], mlp_layers[-1], bias=False)

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


class STMFF(nn.Module):
    def __init__(self, mlp_layers, sigma_x=1, sigma_t_1=1, sigma_t_2=10):
        """
        Spatio Temporal Multiscale Fourier Feature Model
        模型说明：
            mlp_layers[0] 是原始的 [x, t] 输入
            [x, t] 经过 Fourier Embedding 变为 H_x 和 H_t_1, H_t_2
            其中 W_x 和 W_t_1, W_t_2 是 Fourier Embedding 的参数
            进而 H_x 和 H_t_1, H_t_2 先后传入同一个网络 (例如 MLP)
            最后将 H_1 = H_x * H_t_1, H_2 = H_x * H_t_2 作为融合特征
            拼接 H = [H_1, H_2] 传给最后的线性层返回 u
        ---
        参数说明：
            mlp_layers: 神经网络的层结构
                例如 [2, 100, 100, 100, 1]
                但是 mlp_layers[1] 必须是偶数
            sigma_x: 傅里叶特征的参数
                空间域一般取 1, 20, 50, 100, 200, ...
                sigma_x 越大，学习的空间域的频率越高
            sigma_t_1: 傅里叶特征的参数
                时间域一般取 1, ...
                sigma_t_1 越大，学习的时间域的频率越高
            sigma_t_2: 傅里叶特征的参数
                时间域一般取 10, ...
                sigma_t_2 越大，学习的时间域的频率越高
        ---
        示例：
            NN_LAYERS = [2, 100, 100, 100, 1]
            network = STMFF(NN_LAYERS)
        """
        super(STMFF, self).__init__()

        # 傅里叶特征初始化
        self.sigma_x = sigma_x
        self.sigma_t_1 = sigma_t_1
        self.sigma_t_2 = sigma_t_2
        self.W_x = nn.Parameter(torch.randn(1, mlp_layers[1] // 2) * sigma_x, requires_grad=False)
        self.W_t_1 = nn.Parameter(torch.randn(1, mlp_layers[1] // 2) * sigma_t_1, requires_grad=False)
        self.W_t_2 = nn.Parameter(torch.randn(1, mlp_layers[1] // 2) * sigma_t_2, requires_grad=False)
        
        # 网络初始化
        self.model_ = nn.Sequential()
        mlp_layers_ = mlp_layers[1:-1]
        for i in range(len(mlp_layers_) - 1):
            self.model_.add_module(f'fc{i + 1}', nn.Linear(mlp_layers_[i], mlp_layers_[i + 1], bias=True))
            self.model_.add_module(f'act{i + 1}', nn.Tanh())
        self.last_layer_ = nn.Linear(2 * mlp_layers_[-1], mlp_layers[-1], bias=False)

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
