'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 09:24:33
LastEditTime: 2025-02-08 09:37:45
'''
import torch.nn as nn
import torch.nn.init as init


def init_network_weights(module, init_type='kaiming_normal'):
    if isinstance(module, nn.Linear):
        if init_type == 'kaiming_normal':
            # Kaiming Normal 初始化
            init.kaiming_normal_(module.weight)
        elif init_type == 'kaiming_uniform':
            # Kaiming Uniform 初始化
            init.kaiming_uniform_(module.weight)
        elif init_type == 'xavier_normal':
            # Xavier Normal 初始化
            init.xavier_normal_(module.weight)
        elif init_type == 'xavier_uniform':
            # Xavier Uniform 初始化
            init.xavier_uniform_(module.weight)
        elif init_type == 'normal':
            # 正态分布初始化
            init.normal_(module.weight)
        elif init_type == 'uniform':
            # 均匀分布初始化
            init.uniform_(module.weight)
        elif init_type == 'constant':
            # 常数初始化
            init.constant_(module.weight, 0)
        
        # 偏置初始化为零
        if module.bias is not None:
            init.zeros_(module.bias)


if __name__ == '__main__':
    # 测试示例
    # 构建一个简单的网络
    network = nn.Sequential()
    layer = nn.Sequential()
    layer.add_module('fc1', nn.Linear(2, 5, bias=True))
    layer.add_module('act1', nn.Tanh())
    network.add_module('layer1', layer)
    
    layer = nn.Sequential()
    layer.add_module('fc2', nn.Linear(5, 5, bias=True))
    layer.add_module('act2', nn.Tanh())
    network.add_module('layer2', layer)

    layer = nn.Sequential()
    layer.add_module('fc3', nn.Linear(5, 1, bias=False))
    network.add_module('layer3', layer)

    # 初始化网络权重
    network.apply(lambda module: init_network_weights(module, 'xavier_normal'))
    assert True
