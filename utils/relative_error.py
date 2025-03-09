import numpy as np
import torch

def rl2_error(y_hat, y):
    return np.linalg.norm(y_hat - y) / np.linalg.norm(y)

def relative_error_of_solution(pinn, ref_data, num_sample=None):
    r"""
    Calculate the relative error between the predicted solution of the PINN model and the reference solution.

    Args:
        pinn (object): A Physics-Informed Neural Network (PINN) model object.
        ref_data (tuple): A tuple containing input data X and reference solution sol.
        num_sample (int, optional): The number of samples for random sampling. 
            If provided, the input data will be randomly sampled.

    Returns:
        tuple: A tuple containing the relative error error_u and the predicted solution u_pred.
    """
    pinn.eval()
    device = next(pinn.network_solution.parameters()).device

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

    # If num_sample is set, perform random sampling for fast calculation of the relative l2 error of u
    if num_sample is not None:
        idx = np.random.choice(X.shape[0], num_sample, replace=False)
        X = X[idx]
        sol = sol[idx]

    sol_pred = pinn.net_sol(
        torch.from_numpy(X).float().to(device)
    )

    if sol.shape[1] == 1:
        u = sol.flatten()
    elif sol.shape[1] > 1:
        u = [sol[:, [i]].flatten() for i in range(sol.shape[1])]
    else:
        raise

    if isinstance(sol_pred, (list, tuple)):
        u_pred = [u_pred.detach().cpu().numpy().flatten() for u_pred in sol_pred]
        error_u = [rl2_error(u_pred, u) for u_pred, u in zip(u_pred, u)]
    elif isinstance(sol_pred, torch.Tensor):
        u_pred = sol_pred.detach().cpu().numpy().flatten()
        error_u = rl2_error(u_pred, u)
    else:
        raise

    pinn.train()
    return error_u, u_pred


def relative_error_of_parameter(pinn, ref_data, num_sample=None, column_index=None):
    """Calculate the relative error between the predicted parameters of the PINN model and the reference parameters.

    Args:
        pinn (object): A Physics-Informed Neural Network (PINN) model object.
        ref_data (tuple): A tuple containing input data X and reference parameters param.
        num_sample (int, optional): The number of samples for random sampling. 
            If provided, the input data will be randomly sampled.
        column_index: The index of the column of input data X used for parameter prediction.

    Returns:
        tuple: A tuple containing the relative error error_param and the predicted parameters param_pred.
    """
    pinn.eval()
    device = next(pinn.network_parameter.parameters()).device

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

    # If num_sample is set, perform random sampling for fast calculation of the relative l2 error of u
    if num_sample is not None:
        idx = np.random.choice(X.shape[0], num_sample, replace=False)
        X = X[idx]
        param = param[idx]

    param_pred = pinn.net_param(
        torch.from_numpy(X).float().to(device),
        column_index=column_index
    )

    if param.shape[1] == 1:
        param = param.flatten()
    elif param.shape[1] > 1:
        param = [param[:, [i]].flatten() for i in range(param.shape[1])]
    else:
        raise

    if isinstance(param_pred, (list, tuple)):
        param_pred = [p_pred.detach().cpu().numpy().flatten() for p_pred in param_pred]
        error_param = [rl2_error(p_pred, p) for p_pred, p in zip(param_pred, param)]
    elif isinstance(param_pred, torch.Tensor):
        param_pred = param_pred.detach().cpu().numpy().flatten()
        error_param = rl2_error(param_pred, param)
    else:
        raise

    pinn.train()
    return error_param, param_pred
