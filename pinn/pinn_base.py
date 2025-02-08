'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 13:30:02
LastEditTime: 2025-02-08 13:52:03
'''
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data_dict):
        NotImplementedError

    def grad(self, outputs, inputs, n_order=1):
        """
        求 outputs 对 inputs 的 n 阶导 n>=1
        """
        current_grads = outputs
        for k in range(1, n_order + 1):
            if k == 1:
                current_grads = self._grad(outputs, inputs)
            else:
                current_grads = self._grad(current_grads, inputs)
        return current_grads

    def _grad(self, outputs, inputs):
        return torch.autograd.grad(outputs, inputs, 
                                   grad_outputs=torch.ones_like(outputs), 
                                   create_graph=True)[0]


if __name__ == '__main__':
    # 测试示例
    def f(X):
        # return X[:, [0]] * X[:, [1]] * 2
        return (X[:, [0]]**2 + X[:, [1]]**2) * 2
        # return torch.sum((X)**2, dim=1, keepdim=True)

        # 2 * (x^2 + y^2)
        # 2 * 2 * x    n_order=1
        # 2 * 2        n_order=2
        # 0            n_order=3


    n = 3
    x = torch.tensor(range(1, n+1), dtype=float, requires_grad=True)
    x = x.reshape(n, -1)
    y = torch.tensor(range(1, n+1), dtype=float, requires_grad=True)
    y = y.reshape(n, -1)
    print(x)
    print(y)
    # x = torch.randn(n, requires_grad=True)
    # y = torch.randn(n, requires_grad=True)

    X = torch.cat([x, y], dim=1)
    u = f(X)
    print(u)

    pinn = PINN()

    # 1阶导
    # u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u[:, [0]]), create_graph=True)[0]
    u_x = pinn.grad(u, x, 1)
    print(u_x)

    # 2阶导
    # u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x))[0]
    u_xx = pinn.grad(u, x, 2)
    print(u_xx)

    # n阶导
    n_order = 3
    u_nx = pinn.grad(u, x, n_order=n_order)
    print(u_nx)