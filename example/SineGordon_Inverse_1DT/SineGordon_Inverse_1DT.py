r'''
Descripttion: Example of inverse Sine-Gordon problem with PINN.
    Use FFN as solution network.
    Use gradient-dependent weight for residual loss.
Author: Guanyu
Date: 2025-02-08 18:39:32
LastEditTime: 2025-02-16 14:28:30
'''
import os
import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("../../")

from dataset import Dataset1DT
from pinn import PINNInverse
from network import *
from utils import *
from plotting import *


# --------------
# --- 超参数 ---
# --------------
DATA_DIR = './data'
FIGURE_DIR = './figure'
LOG_DIR = './log'
MODEL_DIR = './model'

DOMAIN = (0, 1, 0, 10)  # (x_min, x_max, t_min, t_max)
N_X_RES = 50
N_T_RES = 40
N_OBS = 1000
N_ITERS = 50000
NN_LAYERS = [2] + [200]*2 + [1]
SUB_NN_LAYERS = [1] + [50]*4 + [1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_RES = 1
W_OBS = 1


# --------------------------------------------
# --- 导入参照解 用于计算 Relative L2 error ---
# --------------------------------------------
init_dir(DATA_DIR, FIGURE_DIR, LOG_DIR, MODEL_DIR)

data = io.loadmat(os.path.join(DATA_DIR, 'SineGordon_Inverse_Sol.mat'))
u = data['u']  # shape (N_t, N_x)
x = data['x'].flatten()
t = data['t'].flatten()
alpha = data['param'].flatten()

u_shape = u.shape
xx, tt = np.meshgrid(x, t)
xx, tt, u = xx.flatten(), tt.flatten(), u.flatten()

X = np.stack([xx, tt], axis=1)


# -------------------------------
# --- 方程数据集 用于训练点采样 ---
# -------------------------------
class Dataset(Dataset1DT):
    def __init__(self, domain):
        super().__init__(domain)

    def external_data(self, n_obs=N_OBS):
        # 读取外部数据
        data = io.loadmat(os.path.join(DATA_DIR, 'SineGordon_Inverse_Sol.mat'))
        u = data['u']  # shape (N_t, N_x)
        x = data['x'].flatten()
        t = data['t'].flatten()

        xx, tt = np.meshgrid(x, t)
        xx, tt, u = xx.flatten(), tt.flatten(), u.flatten()

        # 随机采样若干个观测数据
        idx = np.random.choice(len(u), n_obs, replace=False)
        xx, tt, u = xx[idx], tt[idx], u[idx]
        
        X = np.stack([xx, tt], axis=1)
        u = u.reshape(-1, 1)

        self.data_dict["X_obs"] = X
        self.data_dict["u_obs"] = u

    def custom_update(self, n_x_res=N_X_RES, n_t_res=N_T_RES, n_obs=N_OBS):
        self.interior_grid(n_x=n_x_res, n_t=n_t_res)
        self.external_data(n_obs)


# -----------------
# --- PINN 模型 ---
# -----------------
class PINN(PINNInverse):
    def __init__(self, network_solution, network_parameter, should_normalize=True):
        super().__init__(network_solution, network_parameter, should_normalize)

    def forward(self, data_dict):
        # 读取 data_dict 的数据
        X_res, X_obs, u_obs = data_dict["X_res"], data_dict["X_obs"], data_dict["u_obs"]

        # 计算 point-wise loss
        # 便于后续引入权重策略
        loss_dict = {}
        loss_dict['pw_loss_res'] = self.net_res(X_res) ** 2
        loss_dict['pw_loss_obs'] = (self.net_sol(X_obs) - u_obs) ** 2

        return loss_dict
    
    def net_res(self, X):
        x, t = self.split_X_columns_and_require_grad(X)
        u = self.net_sol([x, t])

        u_tt = self.grad(u, t, 2)
        u_xx = self.grad(u, x, 2)

        alpha = self.net_param(t, column_index=-1)

        res_pred = u_tt - alpha * u_xx + torch.sin(u)
        return res_pred
    
    def net_param_output_transform(self, X, parameter):
        alpha = parameter
        alpha += 1
        return alpha


# ---------------------------------------------------
# --- 初始化训练实例 dataset pinn optimizer logger ---
# ---------------------------------------------------
dataset = Dataset(DOMAIN)

network = STFF1DT(NN_LAYERS, sigma_x=1, sigma_t=5)
sub_network = MLP(SUB_NN_LAYERS)
pinn = PINN(network, sub_network)
pinn.mean, pinn.std = dataset.data_dict['mean'], dataset.data_dict['std']

optimizer = optim.Adam(pinn.network_solution.parameters(), lr=1e-3)
sub_optimizer = optim.Adam(pinn.network_parameter.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1000, verbose=True)
sub_scheduler = optim.lr_scheduler.ReduceLROnPlateau(sub_optimizer, mode='min', factor=0.9, patience=1000, verbose=True)

log_keys = ['iter', 'loss', 'loss_res', 'loss_obs', 'error_u', 'error_alpha']
logger = Logger(LOG_DIR, log_keys, num_iters=N_ITERS, print_interval=1000)

def gradient_dependent_weight(pinn, data_dict, beta=1):
    X = data_dict["X_res"]
    x, t = pinn.split_X_columns_and_require_grad(X)

    param_pred = pinn.net_param(t, column_index=-1)
    gradient_of_param_pred = pinn.grad(param_pred, t, 1)
    gradient_of_param_pred = gradient_of_param_pred.norm(dim=1, keepdim=True)
    pw_w_res = 1 / (beta * gradient_of_param_pred + 1)
    return pw_w_res

# ---------------------------------
# --- 开始训练 打印并保存训练信息 ---
# ---------------------------------
best_loss = np.inf
for it in range(N_ITERS):
    pinn.train()
    pinn.zero_grad()                                        # 清除梯度
    loss_dict = pinn(dataset.data_dict)                     # 计算 point-wise loss
    
    pw_loss_res = loss_dict["pw_loss_res"]                  # 提取 point-wise loss
    pw_loss_obs = loss_dict["pw_loss_obs"]

    """----------------------------------------------------------------
    加权 loss, 优化网络参数
    """
    pw_w_res = gradient_dependent_weight(pinn, dataset.data_dict, beta=1)
    loss_res = torch.mean(pw_w_res * pw_loss_res)
    loss_obs = torch.mean(pw_loss_obs)
    loss = W_RES*loss_res + W_OBS*loss_obs
    
    loss.backward()                                         # 反向传播
    """----------------------------------------------------------------
    """

    """----------------------------------------------------------------
    原始 loss, 记录和判别最优模型
    """
    loss_res = torch.mean(pw_loss_res)
    loss_obs = torch.mean(pw_loss_obs)
    loss = W_RES*loss_res + W_OBS*loss_obs
    """----------------------------------------------------------------
    """

    optimizer.step()                                        # 更新网络参数
    sub_optimizer.step()
    scheduler.step(loss)
    sub_scheduler.step(loss)

    error_u, _ = relative_error_of_solution(pinn, ref_data=(X, u), num_sample=500)
    error_alpha, _ = relative_error_of_parameter(
        pinn, ref_data=(t, alpha), num_sample=50, column_index=-1
    )

    logger.record(                                          # 保存训练信息
        iter=it,                                            # 每隔一定次数自动打印
        loss=loss.item(),
        loss_res=loss_res.item(),
        loss_obs=loss_obs.item(),
        error_u=error_u, 
        error_alpha=error_alpha
    )
    
    if loss.item() < best_loss:                    # 保存最优模型
        model_info = {
            'iter': it,
            'nn_sol_state': pinn.network_solution.state_dict(),
            'nn_param_state': pinn.network_parameter.state_dict(),
            'mean': pinn.mean,
            'std': pinn.std,
        }
        torch.save(model_info, os.path.join(MODEL_DIR, 'model.pth'))
        best_loss = loss.item()

logger.print_elapsed_time()
logger.save()


# -----------------------------------
# --- 导入训练信息 以及最优模型参数 ---
# -----------------------------------
logger.load()

model_info = torch.load(os.path.join(MODEL_DIR, 'model.pth'), map_location=DEVICE)
pinn.network_solution.load_state_dict(model_info['nn_sol_state'])
pinn.network_parameter.load_state_dict(model_info['nn_param_state'])
pinn.mean, pinn.std = model_info['mean'], model_info['std']
pinn.eval()


# ----------------------------------
# --- 可视化 loss error solution ---
# ----------------------------------
plot_loss_from_logger(logger, FIGURE_DIR, show=True)
plot_error_from_logger(logger, FIGURE_DIR, show=True)

error_u, u_pred = relative_error_of_solution(pinn, ref_data=(X, u))
error_alpha, alpha_pred = relative_error_of_parameter(pinn, (t, alpha), column_index=-1)

print('Relative l2 error of u: {:.3e}'.format(error_u))
print('Relative l2 error of alpha: {:.3e}'.format(error_alpha))
with open(os.path.join(LOG_DIR, 'relative_error.txt'), 'w') as f_obj:
    f_obj.write('Relative l2 error of u: {:.3e}\n'.format(error_u))
    f_obj.write('Relative l2 error of alpha: {:.3e}\n'.format(error_alpha))

plot_solution_from_data(
    FIGURE_DIR, 
    x_grid=xx.reshape(u_shape),
    y_grid=tt.reshape(u_shape),
    sol=u.reshape(u_shape),
    sol_pred=u_pred.reshape(u_shape),

    x_label='$x$',
    y_label='$t$',

    x_ticks=np.linspace(0, 1, 5),
    y_ticks=np.linspace(0, 10, 6),
    
    title_left=r'Reference $u(x,t)$',
    title_middle=r'Predicted $u(x,t)$',
    title_right=r'Absolute error'
)

plot_solution_from_data(
    FIGURE_DIR,
    x_grid=t,
    sol=alpha,
    sol_pred=alpha_pred,

    x_label='$t$',
    y_label=r'$\lambda_1$',

    x_ticks=np.linspace(0, 10, 6),
    y_ticks=np.linspace(0, 1.5, 4),

    title_left=r'Parameter $\alpha(t)$',
    figure_name='Alpha_PINN.png',
)

# --------------------------
# --- 保存结果为 mat 文件 ---
# --------------------------
io.savemat(
    os.path.join(DATA_DIR, 'SineGordon_Inverse_Sol_PINN.mat'),
    {'x':x, 't':t, 'u':u_pred, 'param':alpha_pred},
)