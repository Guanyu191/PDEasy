'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 08:45:47
LastEditTime: 2025-02-08 11:07:17
'''
import torch
import torch.nn as nn

from utils.init_network_weights import init_network_weights
from utils.init_network_activation_function import init_network_activation_function


class MLP(nn.Module):
    def __init__(self, nn_layers, act_type='tanh', init_type='kaiming_normal'):
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

    def forward(self, X):
        return self.model(X)


class ModifiedMLP(nn.Module):
    def __init__(self, nn_layers, act_type='tanh', init_type='kaiming_normal'):
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

    def forward(self, X):
        u = self.encoder_u(X)
        v = self.encoder_v(X)

        for i in range(len(self.model) - 1):
            X = self.model[i](X)
            X = X / 2.
            X = (1 - X) * u + X * v
        return self.model[-1](X)
