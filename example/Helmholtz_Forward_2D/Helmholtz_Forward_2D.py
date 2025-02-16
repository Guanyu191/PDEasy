r'''
Descripttion: Example for Helmholtz equation with PINN.
Author: Guanyu
Date: 2025-02-08 18:39:32
LastEditTime: 2025-02-09 17:21:26
'''
import os
import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("../../")

from dataset import Dataset2D
from pinn import PINNForward
from network import MFF2D
from utils import *
from plotting import *


# --------------
# --- 超参数 ---
# --------------
DATA_DIR = './data'
FIGURE_DIR = './figure'
LOG_DIR = './log'
MODEL_DIR = './model'

DOMAIN = (-1, 1, -1, 1)  # (x_min, x_max, y_min, y_max)
N_RES = 2000
N_BCS = 200
N_ITERS = 10000
NN_LAYERS = [2] + [50]*4 + [1]


# --------------------------------------------
# --- 导入参照解 用于计算 Relative L2 error ---
# --------------------------------------------
init_dir(DATA_DIR, FIGURE_DIR, LOG_DIR, MODEL_DIR)

data = io.loadmat(os.path.join(DATA_DIR, 'Helmholtz_Sol.mat'))
u = data['u']  # shape (N_y, N_x)
x = data['x'].flatten()
y = data['y'].flatten()

u_shape = u.shape
xx, yy = np.meshgrid(x, y)
xx, yy, u = xx.flatten(), yy.flatten(), u.flatten()

X = np.stack([xx, yy], axis=1)


# -------------------------------
# --- 方程数据集 用于训练点采样 ---
# -------------------------------
class Dataset(Dataset2D):
    def __init__(self, domain):
        super().__init__(domain)

    def custom_update(self, n_res=N_RES, n_bcs=N_BCS):
        self.interior_random(n_res)
        self.boundary_random(n_bcs)


# -----------------
# --- PINN 模型 ---
# -----------------
class PINN(PINNForward):
    def __init__(self, network_solution, should_normalize=True):
        super().__init__(network_solution, should_normalize)

    def forward(self, data_dict):
        # 读取 data_dict 的数据
        X_res, X_bcs = data_dict["X_res"], data_dict["X_bcs"]

        # 计算 point-wise loss
        # 便于后续引入权重策略
        loss_dict = {}
        loss_dict['pw_loss_res'] = self.net_res(X_res) ** 2
        loss_dict['pw_loss_bcs'] = self.net_bcs(X_bcs) ** 2

        return loss_dict
    
    def net_res(self, X):
        columns = self.split_X_columns_and_require_grad(X)
        x, y = columns
        u = self.net_sol([x, y])

        u_xx = self.grad(u, x, 2)
        u_yy = self.grad(u, y, 2)

        a1, a2, k, w = 1, 4, 1, 1
        q = -(a1 * torch.pi) ** 2 * torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y)
        q += -(a2 * torch.pi) ** 2 * torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y)
        q += k ** 2 * torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y) * w
        res_pred = u_xx + u_yy + k**2 * u - q
        return res_pred
    
    def net_bcs(self, X):
        u = self.net_sol(X)
        bcs_pred = u - 0
        return bcs_pred


# ---------------------------------------------------
# --- 初始化训练实例 dataset pinn optimizer logger ---
# ---------------------------------------------------
dataset = Dataset(DOMAIN)

network = MFF2D(NN_LAYERS, sigma_x_2=5, sigma_y_2=10)
pinn = PINN(network)
pinn.mean, pinn.std = dataset.data_dict['mean'], dataset.data_dict['std']

optimizer = optim.Adam(pinn.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=500, verbose=True)

log_keys = ['iter', 'loss', 'loss_res', 'loss_bcs', 'error_u']
logger = Logger(LOG_DIR, log_keys, num_iters=N_ITERS, print_interval=100)

# ---------------------------------
# --- 开始训练 打印并保存训练信息 ---
# ---------------------------------
best_loss = np.inf
for it in range(N_ITERS):
    pinn.zero_grad()                                        # 清除梯度
    loss_dict = pinn(dataset.data_dict)                     # 计算 point-wise loss
    
    pw_loss_res = loss_dict["pw_loss_res"]                  # 提取 point-wise loss
    pw_loss_bcs = loss_dict["pw_loss_bcs"]
    
    loss_res = torch.mean(pw_loss_res)                      # 计算 loss
    loss_bcs = torch.mean(pw_loss_bcs)
        
    loss = loss_res + loss_bcs
    
    loss.backward()                                         # 反向传播    
    optimizer.step()                                        # 更新网络参数
    scheduler.step(loss)                                    # 调整学习率

    error_u, _ = relative_error_of_solution(pinn, ref_data=(X, u), num_sample=500)

    logger.record(                                          # 保存训练信息
        iter=it,                                            # 每隔一定次数自动打印
        loss=loss.item(),
        loss_res=loss_res.item(),
        loss_bcs=loss_bcs.item(),
        error_u=error_u
    )
    
    if it % 100 == 0:
        dataset.update()
    
    if loss.item() < best_loss:                             # 保存最优模型
        model_info = {
            'iter': it,
            'nn_sol_state': pinn.network_solution.state_dict(),
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

model_info = torch.load(os.path.join(MODEL_DIR, 'model.pth'))
pinn.network_solution.load_state_dict(model_info['nn_sol_state'])
pinn.mean, pinn.std = model_info['mean'], model_info['std']
pinn.eval()


# ----------------------------------
# --- 可视化 loss error solution ---
# ----------------------------------
plot_loss_from_logger(logger, FIGURE_DIR, show=True)
plot_error_from_logger(logger, FIGURE_DIR, show=True)

error_u, u_pred = relative_error_of_solution(pinn, ref_data=(X, u))

print('Relative l2 error of u: {:.3e}'.format(error_u))
with open(os.path.join(LOG_DIR, 'relative_error.txt'), 'w') as f_obj:
    f_obj.write('Relative l2 error of u: {:.3e}\n'.format(error_u))

plot_solution_from_data(
    FIGURE_DIR, 
    x_grid=xx.reshape(u_shape),
    y_grid=yy.reshape(u_shape),
    sol=u.reshape(u_shape),
    sol_pred=u_pred.reshape(u_shape),

    x_label='$x$',
    y_label='$y$',

    x_ticks=np.linspace(-1, 1, 5),
    y_ticks=np.linspace(-1, 1, 5),
    
    title_left=r'Reference $u(x,y)$',
    title_middle=r'Predicted $u(x,y)$',
    title_right=r'Absolute error'
)


# --------------------------
# --- 保存结果为 mat 文件 ---
# --------------------------
io.savemat(
    os.path.join(DATA_DIR, 'Helmholtz_Sol_PINN.mat'),
    {'x':x, 'y':y, 'u':u_pred}
)