'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 18:39:32
LastEditTime: 2025-02-11 17:58:27
'''
import os
import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("../../")

from dataset.rectangle import Dataset2DT
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

DOMAIN = (0, 1, 0, 1, 0, 1)  # (x_min, x_max, y_min, y_max, t_min, t_max)
N_RES = 5000
N_OBS = 5000
N_ITERS = 20000
NN_LAYERS = [3] + [80]*4 + [3]
SUB_NN_LAYERS = [1] + [40]*4 + [1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_RES = 1
W_OBS = 100


# --------------------------------------------
# --- 导入参照解 用于计算 Relative L2 error ---
# --------------------------------------------
init_dir(DATA_DIR, FIGURE_DIR, LOG_DIR, MODEL_DIR)

data = io.loadmat(os.path.join(DATA_DIR, 'NavierStokes_Inverse_Sol.mat'))
u = data['u']  # shape (N_y, N_x, N_t)
v = data['v']
p = data['p']
x = data['x'].flatten()
y = data['y'].flatten()
t = data['t'].flatten()
nu = data['nu'].flatten()

u_shape = u.shape
xxx, yyy, ttt = np.meshgrid(x, y, t)
xxx, yyy, ttt = xxx.flatten(), yyy.flatten(), ttt.flatten()
u, v, p = u.flatten(), v.flatten(), p.flatten()

X = np.stack([xxx, yyy, ttt], axis=1)


# -------------------------------
# --- 方程数据集 用于训练点采样 ---
# -------------------------------
class Dataset(Dataset2DT):
    def __init__(self, domain):
        super().__init__(domain)

    def external_data(self, n_obs=N_OBS):
        # 读取外部数据
        data = io.loadmat(os.path.join(DATA_DIR, 'NavierStokes_Inverse_Sol.mat'))
        u = data['u']  # shape (N_y, N_x, N_t)
        v = data['v']
        x = data['x'].flatten()
        y = data['y'].flatten()
        t = data['t'].flatten()

        xxx, yyy, ttt = np.meshgrid(x, y, t)
        xxx, yyy, ttt = xxx.flatten(), yyy.flatten(), ttt.flatten()
        u, v = u.flatten(), v.flatten()

        X = np.stack([xxx, yyy, ttt], axis=1)

        # 随机采样若干个观测数据
        assert len(u) == len(xxx)
        idx = np.random.choice(len(u), n_obs, replace=False)
        xxx, yyy, ttt = xxx[idx], yyy[idx], ttt[idx]
        u, v = u[idx], v[idx]
        
        X = np.stack([xxx, yyy, ttt], axis=1)
        u = u.reshape(-1, 1)
        v = v.reshape(-1, 1)

        self.data_dict['X_obs'] = X
        self.data_dict['u_obs'] = u
        self.data_dict['v_obs'] = v

    def custom_update(self, n_res=N_RES, n_obs=N_OBS):
        self.interior_random(n_res)
        self.external_data(n_obs)


# -----------------
# --- PINN 模型 ---
# -----------------
class PINN(PINNInverse):
    def __init__(self, network_solution, network_parameter, should_normalize=True):
        super().__init__(network_solution, network_parameter, should_normalize)

    def forward(self, data_dict):
        # 读取 data_dict 的数据
        X_res, X_obs = data_dict["X_res"], data_dict["X_obs"]
        u_obs, v_obs = data_dict['u_obs'], data_dict['v_obs']

        # 计算 point-wise loss
        # 便于后续引入权重策略
        loss_dict = {}

        u_res_pred, v_res_pred = self.net_res(X_res)
        u_obs_pred, v_obs_pred, _ = self.net_sol(X_obs)

        loss_dict['pw_loss_u_res'] = u_res_pred ** 2
        loss_dict['pw_loss_v_res'] = v_res_pred ** 2
        loss_dict['pw_loss_u_obs'] = (u_obs_pred - u_obs) ** 2
        loss_dict['pw_loss_v_obs'] = (v_obs_pred - v_obs) ** 2

        return loss_dict
    
    def net_res(self, X):
        x, y, t = self.split_X_columns_and_require_grad(X)
        u, v, p = self.net_sol([x, y, t])

        u_t = self.grad(u, t, 1)
        u_x = self.grad(u, x, 1)
        u_y = self.grad(u, y, 1)
        p_x = self.grad(p, x, 1)
        u_xx = self.grad(u, x, 2)
        u_yy = self.grad(u, y, 2)

        v_t = self.grad(v, t, 1)
        v_x = self.grad(v, x, 1)
        v_y = self.grad(v, y, 1)
        p_y = self.grad(p, y, 1)
        v_xx = self.grad(v, x, 2)
        v_yy = self.grad(v, y, 2)

        nu = self.net_param(t, column_index=-1)

        # 控制方程
        u_res_pred = u_t + (u*u_x + v*u_y) + p_x - nu * (u_xx + u_yy)
        v_res_pred = v_t + (u*v_x + v*v_y) + p_y - nu * (v_xx + v_yy)
        return (u_res_pred, v_res_pred)


# ---------------------------------------------------
# --- 初始化训练实例 dataset pinn optimizer logger ---
# ---------------------------------------------------
dataset = Dataset(DOMAIN)

network = MLP(NN_LAYERS)
sub_network = MLP(SUB_NN_LAYERS)
pinn = PINN(network, sub_network)
pinn.mean, pinn.std = dataset.data_dict['mean'], dataset.data_dict['std']

optimizer = optim.Adam(pinn.network_solution.parameters(), lr=1e-3)
sub_optimizer = optim.Adam(pinn.network_parameter.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1000, verbose=True)
sub_scheduler = optim.lr_scheduler.ReduceLROnPlateau(sub_optimizer, mode='min', factor=0.9, patience=1000, verbose=True)

log_keys = ['iter', 'loss', 'loss_u_res', 'loss_v_res', 'loss_u_obs', 'loss_v_obs', 
            'error_u', 'error_v', 'error_nu']
logger = Logger(LOG_DIR, log_keys, num_iters=N_ITERS, print_interval=100)

# ---------------------------------
# --- 开始训练 打印并保存训练信息 ---
# ---------------------------------
best_loss = np.inf
for it in range(N_ITERS):
    pinn.train()
    pinn.zero_grad()                                        # 清除梯度
    loss_dict = pinn(dataset.data_dict)                     # 计算 point-wise loss
    
    pw_loss_u_res = loss_dict["pw_loss_u_res"]              # 提取 point-wise loss
    pw_loss_v_res = loss_dict["pw_loss_v_res"]
    pw_loss_u_obs = loss_dict["pw_loss_u_obs"]
    pw_loss_v_obs = loss_dict["pw_loss_v_obs"]
    
    loss_u_res = torch.mean(pw_loss_u_res)                  # 计算 loss
    loss_v_res = torch.mean(pw_loss_v_res)
    loss_u_obs = torch.mean(pw_loss_u_obs)
    loss_v_obs = torch.mean(pw_loss_v_obs)
        
    loss = W_RES*(loss_u_res + loss_v_res) + W_OBS*(loss_u_obs + loss_v_obs)
    
    loss.backward()                                         # 反向传播    
    optimizer.step()                                        # 更新网络参数
    sub_optimizer.step()

    scheduler.step(loss)
    sub_scheduler.step(loss)

    error_sol, _ = relative_error_of_solution(              # 抽样计算相对误差
        pinn, ref_data=(X, (u, v, p)), num_sample=500
    )
    error_u, error_v, _ = error_sol
    error_nu, _ = relative_error_of_parameter(
        pinn, ref_data=(t, nu), num_sample=50, column_index=-1
    )

    logger.record(                                          # 保存训练信息
        iter=it,                                            # 每隔一定次数自动打印
        loss=loss.item(),
        loss_u_res=loss_u_res.item(),
        loss_v_res=loss_v_res.item(),
        loss_u_obs=loss_u_obs.item(),
        loss_v_obs=loss_v_obs.item(),
        error_u=error_u, 
        error_v=error_v,
        error_nu=error_nu
    )
    
    if loss.item() < best_loss:                             # 保存最优模型
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

error_sol, sol_pred = relative_error_of_solution(pinn, ref_data=(X, (u, v, p)))
error_u, error_v, _ = error_sol
u_pred, v_pred, _ = sol_pred

error_nu, nu_pred = relative_error_of_parameter(pinn, (t, nu), column_index=-1)

print('Relative l2 error of u: {:.3e}'.format(error_u))
print('Relative l2 error of v: {:.3e}'.format(error_v))
print('Relative l2 error of nu: {:.3e}'.format(error_nu))
with open(os.path.join(LOG_DIR, 'relative_error.txt'), 'w') as f_obj:
    f_obj.write('Relative l2 error of u: {:.3e}\n'.format(error_u))
    f_obj.write('Relative l2 error of v: {:.3e}\n'.format(error_v))
    f_obj.write('Relative l2 error of nu: {:.3e}\n'.format(error_nu))

xxx, yyy, ttt = xxx.reshape(u_shape), yyy.reshape(u_shape), ttt.reshape(u_shape)
u, v = u.reshape(u_shape), v.reshape(u_shape)
u_pred, v_pred = u_pred.reshape(u_shape), v_pred.reshape(u_shape)

snap = -1
xx, yy = xxx[:, :, snap], yyy[:, :, snap]
u, v = u[:, :, snap], v[:, :, snap]
u_pred, v_pred = u_pred[:, :, snap], v_pred[:, :, snap]

plot_solution_from_data(
    FIGURE_DIR, 
    x_grid=xx,
    y_grid=yy,
    sol=u,
    sol_pred=u_pred,

    x_label='$x$',
    y_label='$y$',

    x_ticks=np.linspace(0, 1, 5),
    y_ticks=np.linspace(0, 1, 5),
    
    title_left=r'Reference $u(x, y, -1)$',
    title_middle=r'Predicted $u(x, y, -1)$',
    title_right=r'Absolute error',
    figure_name='Sol_U_PINN.png',
)

plot_solution_from_data(
    FIGURE_DIR, 
    x_grid=xx,
    y_grid=yy,
    sol=v,
    sol_pred=v_pred,

    x_label='$x$',
    y_label='$y$',

    x_ticks=np.linspace(0, 1, 5),
    y_ticks=np.linspace(0, 1, 5),
    
    title_left=r'Reference $v(x, y, -1)$',
    title_middle=r'Predicted $v(x, y, -1)$',
    title_right=r'Absolute error',
    figure_name='Sol_V_PINN.png',
)

plot_solution_from_data(
    FIGURE_DIR,
    x_grid=t,
    sol=nu,
    sol_pred=nu_pred,

    x_label='$t$',
    y_label=r'$\nu$',

    x_ticks=np.linspace(0, 1, 5),
    y_ticks=np.linspace(0.01, 0.07, 4),

    title_left=r'Parameter $\nu(t)$',
    figure_name='Param_Nu_PINN.png',
)

# --------------------------
# --- 保存结果为 mat 文件 ---
# --------------------------
io.savemat(
    os.path.join(DATA_DIR, 'NavierStokes_Inverse_Sol_PINN.mat'),
    {'x':x, 'y':y, 't':t, 'u':u_pred, 'v':v_pred, 'nu':nu_pred},
)