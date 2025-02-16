r"""PINN 基类, 实现便捷的求导功能.

该模块实现了 PINN 最核心的求导功能, 用以计算 residual loss, 并为了简化求导的写法,
附带实现了一些列拆分与拼接的方法, 使得求导的写法更加简洁.

基于该基类实现的 PINN, 用户可以简单的通过 `self.split_X_columns_and_require_grad(X)`
来获得网络输入的坐标, 并要求梯度. 然后通过 `self.grad(u, x, 2)` 来计算导数.
而不再需要调用 `torch.autograd.grad` 计算, 极大地简化求导的写法.

Example::
    >>> class PINNForward(_PINN):
    >>>     ...
    >>>
    >>> class PINN(PINNForward):
    >>>     ...
    >>>     def net_res(self, X):
    >>>         x, t = self.split_X_columns_and_require_grad(X)
    >>>         u = self.net_sol([x, t])
    >>>         u_t = self.grad(u, t, 1)
    >>>         u_xx = self.grad(u, x, 2)
    >>>         return u_t + u * u_x - 0.01 / torch.pi * u_xx
"""

from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn

from torch import Tensor


class _PINN(nn.Module):
    def __init__(self):
        super(_PINN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data_dict:Dict[str, Tensor]) -> Dict[str, Tensor]:
        r"""需要用户重写的 PINN 前向传播方法.

        通过 data_dict 读取 PINN 需要的输入, 并计算各项 point-wise loss, 以 loss_dict 返回.

        loss_dict 的 key 形如 'pw_loss_res', value 则是具体数据, 为 Tensor 数据类型,
        point-wise loss 在传递出去之后可以进一步结合权重算法等, 优化最终的 loss.

        Args:
            data_dict (Dict[str, Tensor]): 数据字典，用于 PINN 的前向传播. 
                其中 key 是数据名称, 例如 'X_res', 而 value 是具体的数据.

        Raises:
            NotImplementedError: 提示用户重写该方法.

        Returns:
            Dict[str, Tensor]: loss_dict, 损失字典, 存储了 PINN 的前向传播结果.
        """
        raise NotImplementedError

    def grad(self, outputs:Tensor, inputs:Tensor, n_order:int = 1) -> Tensor:
        """求 outputs 对 inputs 的 n 阶导.

        导数的阶数 n_order 可以很高 (>=5), 大幅度简化 PINN 中求导的写法.

        outputs 和 inputs 必须是若干行且 1 列的, 表示单个变量.
        二者的计算图关系必须正确.

        Args:
            outputs (Tensor): 若干行, 1 列, Tensor.
            inputs (Tensor): 若干行, 1 列, Tensor.
            n_order (int, optional): >= 1, int. Defaults to 1.

        Returns:
            Tensor: gradient of outputs with respect to inputs. 若干行, 1 列.
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
    
    def split_columns(self, X:Tensor) -> Union[Tensor, List[Tensor]]:
        """对 Tensor 按列拆分.

        X 的维度必须是 2 维, 行数任意. 
        若 X 只有 1 列, 则返回原本的 X (Tensor), 若有 >=2 列, 则返回拆分后的 List[Tensor].        
        拆分后的 List[Tensor] 中的每个 Tensor 都是 1 列, 维数是 2.

        Args:
            X (Tensor): 需要拆分的 Tensor, 若干行, 若干列.

        Returns:
            Union[Tensor, List[Tensor]]: 原 1 列的 Tensor 或拆分后的 List[Tensor].
        """
        # 对于有多列的 X，逐列拆分
        num_columns = X.shape[1]
        if num_columns == 1:
            pass
        elif num_columns > 1:
            X = [X[:, [i]] for i in range(num_columns)]
        else:
            raise
        return X
    
    def cat_columns(self, X:List[Tensor]) -> Tensor:
        """对 List[Tensor] 按列拼接.

        List[Tensor] 中的每个 Tensor 都是 1 列, 维数是 2.
        若 List[Tensor] 只有 1 个 Tensor, 则返回原本的 Tensor, 
        若有 >=2 个 Tensor, 则返回拼接后的 Tensor.

        Args:
            X (List[Tensor]): 需要拼接的 List[Tensor], 若干行, 若干列.

        Returns:
            Tensor: 原 1 列的 Tensor 或拼接后的 Tensor.
        """
        # 对于有多列的 X，逐列拼接
        num_columns = len(X)
        if num_columns == 1:
            pass
        elif num_columns > 1:
            X = torch.cat(X, dim=1)
        else:
            raise
        return X
    
    def split_X_columns_and_require_grad(self, X:Tensor) -> Union[Tensor, List[Tensor]]:
        """对 Tensor 按列拆分, 并对每个 Tensor 都 requires_grad_.

        X 的维度必须是 2 维, 行数任意.
        若 X 只有 1 列, 则返回原本的 X (Tensor), 
        若有 >=2 列, 则返回拆分后的 List[Tensor].
        拆分后的 List[Tensor] 中的每个 Tensor 都是 1 列, 维数是 2.

        Args:
            X (Tensor): 需要拆分并 requires_grad_ 的 Tensor, 若干行, 若干列.

        Returns:
            Union[Tensor, List[Tensor]]: 原 1 列的 Tensor 或拆分后的 List[Tensor].
        """
        # 对于有多列的 X，逐列 requires_grad_ 以便求偏导
        num_columns = X.shape[1]
        if num_columns == 1:
            X.requires_grad_(True)
        elif num_columns > 1:
            X = [X[:, [i]].requires_grad_(True) for i in range(num_columns)]
        else:
            raise
        return X
