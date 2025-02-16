r"""PINN 正问题需要继承该类.

PINN 正问题需要继承该类, 并重写 `forward`, `net_res`, `net_bcs`, `net_ics` 方法.
如果用户有需要对网络输出做变换, 可以通过重写 `net_sol_output_transform` 方法实现.

其中:
    - `forward` 方法是 PINN 正问题的前向传播方法, 用于计算 PDE 的残差和边界条件.
    - `net_res` 方法用于计算 PDE 的残差.
    - `net_bcs` 方法用于计算边界条件.
    - `net_ics` 方法用于计算初始条件.
    - `net_sol_output_transform` 方法用于对网络输出做变换.

Example::
    >>> class PINN(PINNForward):
    >>>     # 以 Burgers 方程为例, 重写 forward, net_res, net_bcs, net_ics 方法
    >>>     def __init__(self, network_solution, should_normalize=True):
    >>>         super().__init__(network_solution, should_normalize)
    >>> 
    >>>     def forward(self, data_dict):
    >>>         # 读取 data_dict 的数据
    >>>         X_res, X_bcs, X_ics = data_dict["X_res"], data_dict["X_bcs"], data_dict["X_ics"]
    >>> 
    >>>         # 计算 point-wise loss
    >>>         # 便于后续引入权重策略
    >>>         loss_dict = {}
    >>>         loss_dict['pw_loss_res'] = self.net_res(X_res) ** 2
    >>>         loss_dict['pw_loss_bcs'] = self.net_bcs(X_bcs) ** 2
    >>>         loss_dict['pw_loss_ics'] = self.net_ics(X_ics) ** 2
    >>> 
    >>>         return loss_dict
    >>>     
    >>>     def net_res(self, X):
    >>>         columns = self.split_X_columns_and_require_grad(X)
    >>>         x, t = columns
    >>>         u = self.net_sol([x, t])
    >>> 
    >>>         u_x = self.grad(u, x, 1)
    >>>         u_t = self.grad(u, t, 1)
    >>>         u_xx = self.grad(u, x, 2)
    >>>         res_pred = u_t + u * u_x - (0.01 / torch.pi) * u_xx
    >>>         return res_pred
    >>>     
    >>>     def net_bcs(self, X):
    >>>         u = self.net_sol(X)
    >>>         bcs_pred = u - 0
    >>>         return bcs_pred
    >>>     
    >>>     def net_ics(self, X):
    >>>         u = self.net_sol(X)
    >>>         ics_pred = u + torch.sin(torch.pi * X[:, [0]])
    >>>         return ics_pred 
"""
import torch
from pinn.pinn_base import _PINN

from torch import Tensor
from torch.nn import Module
from typing import List, Union


class PINNForward(_PINN):
    def __init__(self, network_solution: Module, should_normalize: bool = True) -> None:
        r"""初始化 PINN 的网络和相关参数.

        Args:
            network_solution (Module): 输出 PDE 解的网络.
            should_normalize (bool, optional): 是否对输入坐标标准化. Defaults to True.
        """

        super(PINNForward, self).__init__()
        self.network_solution = network_solution
        self.should_normalize = should_normalize
        self.mean = None
        self.std = None
        
        self.to(self.device)

    def net_sol(self, X: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        r"""根据输入的坐标 X, 输出 PDE 的解.

        该方法封装了从输入坐标 X 到输出解的完整过程, 包括: 
            1. 对输入做标准化. X = (X - mean) / std
            2. 调用 `self.network_solution` 得到网络输出
            3. 调用 `self.net_sol_output_transform` 得到最终的输出

        需要获得 PDE 的解时, 请调用该方法 (`self.net_sol`), 而不是直接调用 `self.network_soluton`.
        
        该方法的输入可以接受 Tensor 或 List[Tensor], 建议使用 List[Tensor], 
        并将时间坐标放在最后一列, 即 [x, t] 或 [x, y, t].

        输出如果是 1 维, 则返回 Tensor, 如果是多维, 则返回 List[Tensor]. 

        Args:
            X (Union[Tensor, List[Tensor]]): 未标准化的坐标张量.

        Returns:
            Union[Tensor, List[Tensor]]: PINN 求解结果.

        Example::
            >>> u = self.net_sol([x, t])  # for 1D outputs
            >>> u, v, p = self.net_sol([x, y, t])  # for 3D outputs
        """
        if isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, (list, tuple)):
            X = self.cat_columns(X)
        else:
            raise
        
        if self.should_normalize:
            X = (X - self.mean) / self.std
        
        solution = self.network_solution(X)

        X = self.split_columns(X)
        solution = self.split_columns(solution)

        solution = self.net_sol_output_transform(X, solution)

        if isinstance(solution, torch.Tensor):
            solution = self.split_columns(solution)
        elif isinstance(solution, (list, tuple)):
            if solution[0].shape[1] > 1:
                raise
        else:
            raise
        return solution
    
    def net_sol_output_transform(
            self, 
            X: Union[Tensor, List[Tensor]], 
            solution: Union[Tensor, List[Tensor]]
        ) -> Union[Tensor, List[Tensor]]:
        r"""对网络直接输出的解做变换.

        默认不变换, 若需要做输出变换, 需要用户重写该方法.

        常见的变换主要有边界约束, 尺度放缩.

        Args:
            X (Union[Tensor, List[Tensor]]): 网络输入的坐标.
            solution (Union[Tensor, List[Tensor]]): 网络输出的解.

        Returns:
            Union[Tensor, List[Tensor]]: 变换后的解.

        Example::
            >>> class PINN(PINNForward):
            >>>     ...
            >>>     def net_sol_output_transform(self, X, u):
            >>>         # Allen Cahn Equation 的输出变换
            >>>         # 使得解自动满足初始条件和边界条件
            >>>         # x^{2} \cos(\pi x) + t (1 - x^{2}) u
            >>>         u = X[:, [0]]**2 * torch.cos(torch.pi * X[:, [0]]) 
            >>>         u += X[:, [1]] * (1 - X[:, [0]]**2) * u
            >>>         return u
        """
        return solution
        
    def net_res(self, X: Tensor) -> Tensor:
        r"""计算 PDE 的方程 point-wise loss.

        注意, 在重写该方法时, 首先要调用 `self.split_X_columns_and_require_grad(X)`,
        以获得网络输入的坐标, 并要求梯度. 然后根据具体的 PDE 形式, 计算 point-wise loss.

        Args:
            X (Tensor): 网络输入的坐标 (内部点).

        Raises:
            NotImplementedError: if 没有重写.

        Returns:
            Tensor: PDE 的方程 point-wise loss.

        Example::
            >>> class PINN(PINNForward):
            >>>     ...
            >>>     def net_res(self, X):
            >>>         x, t = self.split_X_columns_and_require_grad(X)
            >>>         u = self.net_sol([x, t])
            >>>         u_t = self.grad(u, t, 1)
            >>>         u_xx = self.grad(u, x, 2)
            >>>         res_pred = u_t - 0.001 * u_xx - 5 * (u - u**3)
            >>>         return res_pred
        """
        raise NotImplementedError

    def net_bcs(self, X: Tensor) -> Tensor:
        r"""计算 PDE 的边界条件 point-wise loss.

        Args:
            X (Tensor): 网络输入的坐标 (边界点).

        Raises:
            NotImplementedError: if 没有重写.

        Returns:
            Tensor: PDE 的边界条件 point-wise loss.

        Example::
            >>> class PINN(PINNForward):
            >>>     ...
            >>>     def net_bcs(self, X):
            >>>         u = self.net_sol(X)
            >>>         u_left = u[X[:, [0]] == -1]
            >>>         u_right = u[X[:, [0]] == 1]
            >>>         bcs_pred = u_left - u_right
            >>>         return bcs_pred
        """
        raise NotImplementedError

    def net_ics(self, X: Tensor) -> Tensor:
        r"""计算 PDE 的初始条件 point-wise loss.

        Args:
            X (Tensor): 网络输入的坐标 (初始点).

        Raises:
            NotImplementedError: if 没有重写.

        Returns:
            Tensor: PDE 的初始条件 point-wise loss.

        Example::
            >>> class PINN(PINNForward):
            >>>     ...
            >>>     def net_ics(self, X):
            >>>         u = self.net_sol(X)
            >>>         ics_pred = u - X[:, [0]]**2 * torch.cos(torch.pi * X[:, [0]])
            >>>         return ics_pred
        """
        raise NotImplementedError
