'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 14:20:45
LastEditTime: 2025-02-11 16:26:55
'''
import numpy as np
import torch

def rl2_error(y_hat, y):
    return np.linalg.norm(y_hat - y) / np.linalg.norm(y)

def relative_error_of_solution(pinn, ref_data, num_sample=None):
    """
    计算相对误差
    """
    pinn.eval()
    device = next(pinn.network_solution.parameters()).device

    # 这里的 X 一定是要整体的形式，用以直接喂给网络
    # 这里的 sol 一定要是整体的形式，为了先采样 num_sample
    X, sol = ref_data
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    elif isinstance(X, (list, tuple)):
        if X[0].ndim == 1:
            X = np.stack(X, axis=1)
        elif X[0].ndim == 2:
            X = np.concatenate(X, axis=1)
    else:
        raise

    if isinstance(sol, np.ndarray):
        if sol.ndim == 1:
            sol = sol.reshape(-1, 1)
    elif isinstance(sol, (list, tuple)):
        if sol[0].ndim == 1:
            sol = np.stack(sol, axis=1)
        elif sol[0].ndim == 2:
            sol = np.concatenate(sol, axis=1)
        else:
            raise
    else:
        raise

    # 如果设置了 num_sample 则随机采样以便快速计算 u 的 relative l2 error
    if num_sample is not None:
        idx = np.random.choice(X.shape[0], num_sample, replace=False)
        X = X[idx]
        sol = sol[idx]

    # 喂 X 给网络，计算 sol_pred
    sol_pred = pinn.net_sol(
        torch.from_numpy(X).float().to(device)
    )

    # 前面已经将 sol 转换为 ndim=2 的 ndarray 形式
    # 下面希望拆分成逐列的形式，用于计算 relative l2 error
    # 如果 sol 只有一列，则直接 flatten
    # 如果 sol 是多列的，则分开 sol 为逐列的形式，再 flatten
    if sol.shape[1] == 1:
        u = sol.flatten()
    elif sol.shape[1] > 1:
        u = [sol[:, [i]].flatten() for i in range(sol.shape[1])]
    else:
        raise

    # 因为 net_sol 一定是以单列的形式返回的
    # 所以，想要知道有几个反演参数要计算，则判断它是否为 list/tuple
    # 如果是，则逐列计算 error，如果不是，则直接计算 error
    if isinstance(sol_pred, (list, tuple)):
        u_pred = [u_pred.detach().cpu().numpy().flatten() for u_pred in sol_pred]
        error_u = [rl2_error(u_pred, u) for u_pred, u in zip(u_pred, u)]
    elif isinstance(sol_pred, torch.Tensor):
        u_pred = sol_pred.detach().cpu().numpy().flatten()
        error_u = rl2_error(u_pred, u)
    else:
        raise

    pinn.train()
    # 返回 u 的误差和预测值
    return error_u, u_pred


def relative_error_of_parameter(pinn, ref_data, num_sample=None, column_index=None):
    pinn.eval()
    device = next(pinn.network_parameter.parameters()).device

    # 这里的 X 一定是要整体的形式，用以直接喂给网络
    # 这里的 param 一定要是整体的形式，为了先采样 num_sample
    X, param = ref_data
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    elif isinstance(X, (list, tuple)):
        if X[0].ndim == 1:
            X = np.stack(X, axis=1)
        elif X[0].ndim == 2:
            X = np.concatenate(X, axis=1)
    else:
        raise

    if isinstance(param, np.ndarray):
        if param.ndim == 1:
            param = param.reshape(-1, 1)
    elif isinstance(param, (list, tuple)):
        if param[0].ndim == 1:
            param = np.stack(param, axis=1)
        elif param[0].ndim == 2:
            param = np.concatenate(param, axis=1)
        else:
            raise
    else:
        raise

    # 如果设置了 num_sample 则随机采样以便快速计算 u 的 relative l2 error
    if num_sample is not None:
        idx = np.random.choice(X.shape[0], num_sample, replace=False)
        X = X[idx]
        param = param[idx]

    # 喂 X 给网络，计算 param_pred
    param_pred = pinn.net_param(
        torch.from_numpy(X).float().to(device),
        column_index=column_index
    )

    # 前面已经将 param 转换为 ndim=2 的 ndarray 形式
    # 下面希望拆分成逐列的形式，用于计算 relative l2 error
    # 如果 param 只有一列，则直接 flatten
    # 如果 param 是多列的，则分开 param 为逐列的形式，再 flatten
    if param.shape[1] == 1:
        param = param.flatten()
    elif param.shape[1] > 1:
        param = [param[:, [i]].flatten() for i in range(param.shape[1])]
    else:
        raise

    # 因为 net_param 一定是以单列的形式返回的
    # 所以，想要知道有几个反演参数要计算，则判断它是否为 list/tuple
    # 如果是，则逐列计算 error，如果不是，则直接计算 error
    if isinstance(param_pred, (list, tuple)):
        param_pred = [p_pred.detach().cpu().numpy().flatten() for p_pred in param_pred]
        error_param = [rl2_error(p_pred, p) for p_pred, p in zip(param_pred, param)]
    elif isinstance(param_pred, torch.Tensor):
        param_pred = param_pred.detach().cpu().numpy().flatten()
        error_param = rl2_error(param_pred, param)
    else:
        raise

    pinn.train()
    # 返回反演参数的误差和预测值
    return error_param, param_pred
