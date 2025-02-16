r"""PINN 反问题需要继承该类.

该类的主要作用是封装 PINN 的结构, 并计算 PDE 的方程 point-wise loss.

PINN 反问题需要继承该类, 并重写 `forward`, `net_res` 方法.
反问题的 `net_res` 不仅需要结合 `net_sol` 得到的解, 还需要结合 `net_param` 得到的参数.
融合二者的输出, 计算 PDE 的 residual point-wise loss.

其中:
    - `forward` 方法用于计算 PDE 的方程, 边界条件, 初始条件 point-wise loss.
    - `net_res` 方法用于计算 PDE 的方程 point-wise loss.

Example::
    >>> class PINN(PINNInverse):
    >>>     def __init__(self, network_solution, network_parameter, should_normalize=True):
    >>>         super().__init__(network_solution, network_parameter, should_normalize)
    >>> 
    >>>     def forward(self, data_dict):
    >>>         # 读取 data_dict 的数据
    >>>         X_res, X_obs, u_obs = data_dict["X_res"], data_dict["X_obs"], data_dict["u_obs"]
    >>> 
    >>>         # 计算 point-wise loss
    >>>         # 便于后续引入权重策略
    >>>         loss_dict = {}
    >>>         loss_dict['pw_loss_res'] = self.net_res(X_res) ** 2
    >>>         loss_dict['pw_loss_obs'] = (self.net_sol(X_obs) - u_obs) ** 2
    >>> 
    >>>         return loss_dict
    >>>     
    >>>     def net_res(self, X):
    >>>         x, t = self.split_X_columns_and_require_grad(X)
    >>>         u = self.net_sol([x, t])
    >>> 
    >>>         u_x = self.grad(u, x, 1)
    >>>         u_t = self.grad(u, t, 1)
    >>>         u_xx = self.grad(u, x, 2)
    >>> 
    >>>         parameter = self.net_param(t, column_index=-1)
    >>>         lam_1, lam_2 = parameter
    >>> 
    >>>         res_pred = u_t - lam_1 * u * u_x - lam_2 * u_xx
    >>>         return res_pred
    >>>     
    >>>     def net_param_output_transform(self, X, parameter):
    >>>         # 作尺度变换
    >>>         lam_1, lam_2 = parameter
    >>>         lam_1 *= 1.
    >>>         lam_2 *= 0.1
    >>>         parameter = [lam_1, lam_2]
    >>>         return parameter
"""
import torch
from pinn.pinn_base import _PINN

from torch import Tensor
from torch.nn import Module
from typing import List, Union


class PINNInverse(_PINN):
    def __init__(
            self, 
            network_solution: Module, 
            network_parameter: Module, 
            should_normalize: bool = True
        ) -> None:
        r"""初始化 PINN 的网络和相关参数.

        Args:
            network_solution (Module): 输出 PDE 的解的网络.
            network_parameter (Module): 输出 PDE 的反演参数的网络.
            should_normalize (bool, optional): 是否对输入坐标做标准化. Defaults to True.
        """
        super(PINNInverse, self).__init__()
        self.network_solution = network_solution
        self.network_parameter = network_parameter
        self.should_normalize = should_normalize
        self.mean = None
        self.std = None
        
        self.to(self.device)

    def net_sol(
            self, 
            X: Union[Tensor, List[Tensor]]
        ) -> Union[Tensor, List[Tensor]]:
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
    
    def net_param(
            self, 
            X: Union[Tensor, List[Tensor]], 
            column_index: int = None
        ) -> Union[Tensor, List[Tensor]]:
        r"""根据输入的坐标 X, 输出 PDE 的反演参数.

        该方法封装了从输入坐标 X 到输出反演参数的完整过程, 包括:
            1. 对输入做标准化. X = (X - mean) / std
            2. 调用 `self.network_parameter` 得到网络输出
            3. 调用 `self.net_param_output_transform` 得到最终的输出
        
        注意, 需要获得 PDE 的解时, 请调用该方法 (`self.net_param`), 而不是直接调用 `self.network_parameter`.

        该方法的输入可以接受 Tensor 或 List[Tensor], 建议使用 List[Tensor],
        并将时间坐标放在最后一列, 即 [x, t] 或 [x, y, t].
        输出如果是 1 维, 则返回 Tensor, 如果是多维, 则返回 List[Tensor].

        Args:
            X (Union[Tensor, List[Tensor]]): _description_
            column_index (int, optional): _description_. Defaults to None.

        Returns:
            Union[Tensor, List[Tensor]]: _description_
        """

        # 当 column_index 为 None 时，X 是完整的
        # 当 column_index 不为 None 时，X 是部分指定列的
        # 例如 net_param(t, column_index=-1) 表示只用到时间 t 预测参数
        
        # 判断 X 的类型，支持 Tensor 和 list/tuple
        # 为了直接输入网络，所以要确保 X 是整体的
        if isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, (list, tuple)):
            X = self.cat_columns(X)
        else:
            raise

        # 标准化
        # 如果没有指定部分列，那么照常标准化
        # 如果指定了部分列，那么就只抽取那些列对应的参数做标准化
        if self.should_normalize:
            if column_index is None:
                X = (X - self.mean) / self.std
            else:
                X = (X - self.mean[column_index]) / self.std[column_index]

        parameter = self.network_parameter(X)

        # 以单列的输出形式给到 output_transform 函数
        X = self.split_columns(X)
        parameter = self.split_columns(parameter)

        # 输出变换
        parameter = self.net_param_output_transform(X, parameter)

        # 确保以单列的输出形式返回
        if isinstance(parameter, torch.Tensor):
            parameter = self.split_columns(parameter)
        elif isinstance(parameter, (list, tuple)):
            pass
        else:
            raise

        return parameter
    
    def net_param_output_transform(
        self, 
        X: Union[Tensor, List[Tensor]], 
        parameter: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, List[Tensor]]:
        r"""对网络输出的反演参数做输出变换.

        该方法对网络输出的反演参数做变换, 使网络更易拟合, 常见的变换为尺度变换.
        例如, 对于 Burgers 方程, 可以对输出乘以一个尺度, 使得网络预测值在 [-1, 1].

        该方法的输入是 Tensor 或 List[Tensor], 对于多变量的 X, parameter, 可以采用
        `x, t = X`, `lam_1, lam_2 = parameter` 来分别获得具体变量.

        该方法的输出是变换后的 parameter, 类型为 Tensor 或 List[Tensor], 对于多列的
        parameter, 尽管可以接受拼接的, 但仍然建议写成不拼接的 `return [lam_1, lam_2]`

        Args:
            X (Union[Tensor, List[Tensor]]): 网络输入的坐标.
            parameter (Union[Tensor, List[Tensor]]): 网络输出的参数 (未变换).

        Returns:
            Union[Tensor, List[Tensor]]: 变换后的参数.

        Example::
            >>> class PINN(PINNForward):
            >>>     ...
            >>>     def net_param_output_transform(self, X, parameter):
            >>>         # 对参数做尺度变换
            >>>         # 使得网络预测值在 [-1, 1] 范围内
            >>>         lam_1, lam_2 = parameter
            >>>         lam_1 = torch.tanh(lam_1)
            >>>         lam_2 = torch.exp(6 * lam_2)
            >>>         return [lam_1, lam_2]
        """
        # 需要用户自行定义，默认无变换
        return parameter
    
    def net_res(self, X: Tensor) -> Tensor:
        r"""计算 PDE 的方程 point-wise loss.

        注意, 在重写该方法时, 首先要调用 `self.split_X_columns_and_require_grad(X)`,
        以获得网络输入的坐标, 并要求梯度. 然后根据具体的 PDE 形式, 计算 point-wise loss.

        注意, 对于反问题, 计算 PDE 内部点的 point-wise loss 时, 需要调用 
        `self.net_param` 来获得参数, 并根据 PDE 具体形式将参数融合进 loss.

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
            >>>         u_x = self.grad(u, x, 1)
            >>>         u_t = self.grad(u, t, 1)
            >>>         u_xx = self.grad(u, x, 2)
            >>>         lam_1, lam_2 = self.net_param(t, column_index=-1)
            >>>         res_pred = u_t - lam_1 * u * u_x - lam_2 * u_xx
            >>>         return res_pred
        """
        raise NotImplementedError
