'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 14:20:45
LastEditTime: 2025-02-10 23:04:53
'''
import numpy as np
import torch

def rl2_error(y_hat, y):
    return np.linalg.norm(y_hat.flatten() - y.flatten()) / np.linalg.norm(y.flatten())

def relative_error_of_solution(pinn, ref_data, num_sample=None):
    """
    计算相对误差
    """
    pinn.eval()
    device = next(pinn.network_solution.parameters()).device

    X, u = ref_data

    # 如果设置了 num_sample 则随机采样以便快速计算 u 的 relative l2 error
    if num_sample is not None:
        idx = np.random.choice(X.shape[0], num_sample, replace=False)
        X = X[idx]
        u = u[idx]

    # 计算 u 的 relative l2 error
    u_pred = pinn.net_sol(
        torch.from_numpy(X).float().to(device)
    )
    u_pred = u_pred.detach().cpu().numpy().flatten()
    error_u = rl2_error(u_pred, u)

    pinn.train()
    
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

    # 分开 param 为逐列的形式，用于计算 relative l2 error
    param = [param[:, [i]] for i in range(param.shape[1])]

    # 因为 net_param 一定是以单列的形式返回的
    # 所以，想要知道有几个反演参数要计算，则判断它是否为 list/tuple，
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
