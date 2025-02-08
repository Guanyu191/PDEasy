'''
Descripttion: test whether the mlp.py is working
Author: Guanyu
Date: 2025-02-08 10:35:46
LastEditTime: 2025-02-08 11:07:44
'''
import torch

import sys
sys.path.append("./")

for path in sys.path:
    print(path)

from network.mlp import MLP



if __name__ == '__main__':
    # 测试示例
    nn_layers = [2, 5, 5, 1]
    model = MLP(nn_layers, 'tanh', 'constant')

    x = torch.tensor([[1., 2.]])
    print(model)
    print(model(x))
    assert True