import torch
import torch.nn as nn

class CosActivation(nn.Module):
    def forward(self, x):
        return torch.cos(2 * torch.pi * x)
    
class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(2 * torch.pi * x)

def init_network_activation_function(name):
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif name.lower() == 'gelu':
        return nn.GELU()
    elif name.lower() == 'selu':
        return nn.SELU()
    elif name.lower() == 'softplus':
        return nn.Softplus()
    elif name.lower() == 'hardtanh':
        return nn.Hardtanh()
    elif name.lower() == 'prelu':
        return nn.PReLU()
    elif name.lower() == 'rrelu':
        return nn.RReLU()
    elif name.lower() == 'elu':
        return nn.ELU()
    elif name.lower() == 'cos':
        return CosActivation()
    elif name.lower() == 'sin':
        return SinActivation()
    elif name.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function: {name}. Supported functions: relu, leakyrelu, tanh, sigmoid, gelu, selu, softplus, hardtanh, prelu, rrelu, elu.")


if __name__ == '__main__':
    activation_func = init_network_activation_function('tanh')
    assert True
