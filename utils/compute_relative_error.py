'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 14:20:45
LastEditTime: 2025-02-08 14:34:58
'''
import numpy as np
import torch

def compute_relative_error(pinn, ref_data, num_sample=None):
    """
    计算相对误差
    """
    pinn.eval()
    device = next(pinn.net_sol.parameters()).device


    X = ref_data['X']
    u = ref_data['u']

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
    error_u = np.linalg.norm(u_pred - u) / np.linalg.norm(u)

    pinn.train()
    
    return error_u, u_pred




