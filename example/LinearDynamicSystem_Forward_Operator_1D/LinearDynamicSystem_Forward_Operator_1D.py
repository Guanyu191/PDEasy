r'''
Descripttion: Example of operator learning on 1D linear dynamic system.
Author: Guanyu
Date: 2025-02-08 18:39:32
LastEditTime: 2025-02-16 16:13:55
'''
import os
import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("../../")

from dataset import Dataset1D
from framework import DeepONetForward
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

DOMAIN = (0, 1)  # (x_min, x_max)

N_BATCH = 100
N_FUNC = 100
N_SENSOR = 100
N_DIM = 1
N_FUNC_TEST = 10 * N_FUNC

N_ITERS = 1000
NN_BRANCH_LAYERS = [N_SENSOR] + [40]*3
NN_TRUNK_LAYERS = [N_DIM] + [40]*2


# --------------------------------------------
# --- 导入参照解 用于计算 Relative L2 error ---
# --------------------------------------------
init_dir(DATA_DIR, FIGURE_DIR, LOG_DIR, MODEL_DIR)

trainset = io.loadmat(os.path.join(DATA_DIR, 'LinearDynamicSystem_Sol_Trainset.mat'))
F_train = trainset['F']  # (N_FUNC * N_LOC, N_SENSOR)
X_train = trainset['X']  # (N_FUNC * N_LOC, N_DIM)
U_train = trainset['U']  # (N_FUNC * N_LOC, 1)
testset = io.loadmat(os.path.join(DATA_DIR, 'LinearDynamicSystem_Sol_Testset.mat'))
F_test = testset['F']
X_test = testset['X']
U_test = testset['U']


# -------------------------------
# --- 方程数据集 用于训练点采样 ---
# -------------------------------
class Dataset(Dataset1D):
    def __init__(self, domain):
        super().__init__(domain)

    def external_data(self):
        trainset = io.loadmat(os.path.join(DATA_DIR, 'LinearDynamicSystem_Sol_Trainset.mat'))
        F_train = trainset['F']  # (N_FUNC * N_LOC, N_SENSOR)
        X_train = trainset['X']  # (N_FUNC * N_LOC, N_DIM)
        U_train = trainset['U']  # (N_FUNC * N_LOC, 1)

        idx = np.random.choice(F_train.shape[0], N_BATCH, replace=False)
        self.data_dict['F'] = F_train[idx]
        self.data_dict['X'] = X_train[idx]
        self.data_dict['U'] = U_train[idx]

    def statistic(self):
        trainset = io.loadmat(os.path.join(DATA_DIR, 'LinearDynamicSystem_Sol_Trainset.mat'))
        F_train = trainset['F']  # (N_FUNC * N_LOC, N_SENSOR)
        X_train = trainset['X']  # (N_FUNC * N_LOC, N_DIM)
        U_train = trainset['U']  # (N_FUNC * N_LOC, 1)

        self.data_dict['F_mean'] = np.mean(F_train, axis=(0, 1), keepdims=True)
        self.data_dict['F_std'] = np.std(F_train, axis=(0, 1), keepdims=True)
        self.data_dict['X_mean'] = np.mean(X_train, axis=0, keepdims=True)
        self.data_dict['X_std'] = np.std(X_train, axis=0, keepdims=True)
        self.data_dict['U_mean'] = np.mean(U_train, axis=0, keepdims=True)
        self.data_dict['U_std'] = np.std(U_train, axis=0, keepdims=True)

    def custom_update(self):
        self.external_data()


# ---------------------
# --- DeepONet 模型 ---
# ---------------------
class DeepONet(DeepONetForward):
    def __init__(self, network_branch, network_trunk, should_normalize=True):
        super().__init__(network_branch, network_trunk, 
                         should_normalize=should_normalize)

    def forward(self, data_dict):
        # 读取 data_dict 的数据
        F, X, U = data_dict["F"], data_dict["X"], data_dict["U"]

        U_pred = self.net_sol(F, X)

        loss_dict = {}
        loss_dict['pw_loss'] = (U_pred - U) ** 2
        return loss_dict


# ---------------------------------------------------
# --- 初始化训练实例 dataset pinn optimizer logger ---
# ---------------------------------------------------
dataset = Dataset(DOMAIN)

network_branch = MLP(NN_BRANCH_LAYERS)
network_trunk = MLP(NN_TRUNK_LAYERS)
deeponet = DeepONet(network_branch, network_trunk)
deeponet.F_mean, deeponet.F_std = dataset.data_dict['F_mean'], dataset.data_dict['F_std']
deeponet.X_mean, deeponet.X_std = dataset.data_dict['X_mean'], dataset.data_dict['X_std']
deeponet.U_mean, deeponet.U_std = dataset.data_dict['U_mean'], dataset.data_dict['U_std']

optimizer = optim.Adam(deeponet.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=1000)

log_keys = ['iter', 'loss_train', 'loss_test']
logger = Logger(LOG_DIR, log_keys, num_iters=N_ITERS, print_interval=100)

F_test = torch.from_numpy(F_test).float().to(deeponet.device)
X_test = torch.from_numpy(X_test).float().to(deeponet.device)
U_test = torch.from_numpy(U_test).float().to(deeponet.device)

# ---------------------------------
# --- 开始训练 打印并保存训练信息 ---
# ---------------------------------
best_loss = np.inf
for it in range(N_ITERS):
    deeponet.zero_grad()                                    # 清除梯度
    loss_dict = deeponet(dataset.data_dict)                 # 计算 point-wise loss
    
    pw_loss = loss_dict["pw_loss"]                          # 提取 point-wise loss

    loss = torch.mean(pw_loss)                              # 计算 loss

    loss.backward()                                         # 反向传播    
    optimizer.step()                                        # 更新网络参数
    scheduler.step(loss)                                    # 调整学习率

    with torch.no_grad():
        idx = torch.randperm(U_test.shape[0])[:100]
        U_teset_pred = deeponet.net_sol(F_test, X_test)
        loss_test = torch.mean((U_teset_pred - U_test) ** 2)

    logger.record(                                          # 保存训练信息
        iter=it,                                            # 每隔一定次数自动打印
        loss_train=loss.item(),
        loss_test=loss_test.item(),
    )
    
    if it % 1000 == 0:
        dataset.update()
                
    if loss.item() < best_loss:                             # 保存最优模型
        model_info = {
            'iter': it,
            'nn_sol_state': deeponet.state_dict(),
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
deeponet.load_state_dict(model_info['nn_sol_state'])
deeponet.eval()


# ----------------------------------
# --- 可视化 loss error solution ---
# ----------------------------------
plot_loss_from_logger(logger, FIGURE_DIR, show=True)

U_test_pred = deeponet.net_sol(F_test, X_test)
U_test = U_test.detach().cpu().numpy()
U_test_pred = U_test_pred.detach().cpu().numpy()
error = np.linalg.norm(U_test_pred - U_test) / np.linalg.norm(U_test)

print('Relative l2 error: {:.3e}'.format(error))
with open(os.path.join(LOG_DIR, 'relative_error.txt'), 'w') as f_obj:
    f_obj.write('Relative l2 error: {:.3e}\n'.format(error))

import matplotlib.pyplot as plt

X_test = X_test.detach().cpu().numpy()

X_test = X_test.reshape(N_FUNC_TEST, -1)
U_test = U_test.reshape(N_FUNC_TEST, -1)
U_test_pred = U_test_pred.reshape(N_FUNC_TEST, -1)

idx = np.random.choice(U_test.shape[0], 3, replace=False)
x_min, x_max = DOMAIN
sol_min, sol_max = U_test.min(), U_test.max()
e = (sol_max - sol_min) * 0.05
x_ticks = np.linspace(x_min, x_max, 5)
y_ticks = np.linspace(np.floor(sol_min), np.ceil(sol_max), 5)

fig = plt.figure(figsize=(18, 5))
axes = fig.subplots(1, 3)

ax = axes[0]
ax.plot(X_test[idx[0]], U_test_pred[idx[0]], 'b-', linewidth=2, label='Predicted')
ax.plot(X_test[idx[0]], U_test[idx[0]], 'r--', linewidth=2, label='Reference')
ax.set_xlim(x_min, x_max)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x)$')
ax.set_title('Sample 1')
ax.set_ylim(sol_min-e, sol_max+e)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_aspect(1./ax.get_data_ratio())
ax.grid()
ax.legend(loc='upper right', fontsize=14)

ax = axes[1]
ax.plot(X_test[idx[1]], U_test_pred[idx[1]], 'b-', linewidth=2, label='Predicted')
ax.plot(X_test[idx[1]], U_test[idx[1]], 'r--', linewidth=2, label='Reference')
ax.set_xlim(x_min, x_max)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x)$')
ax.set_title('Sample 1')
ax.set_ylim(sol_min-e, sol_max+e)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_aspect(1./ax.get_data_ratio())
ax.grid()
ax.legend(loc='upper right', fontsize=14)

ax = axes[2]
ax.plot(X_test[idx[2]], U_test_pred[idx[2]], 'b-', linewidth=2, label='Predicted')
ax.plot(X_test[idx[2]], U_test[idx[2]], 'r--', linewidth=2, label='Reference')
ax.set_xlim(x_min, x_max)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x)$')
ax.set_title('Sample 1')
ax.set_ylim(sol_min-e, sol_max+e)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_aspect(1./ax.get_data_ratio())
ax.grid()
ax.legend(loc='upper right', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'Sol_DeepONet'), dpi=300, bbox_inches='tight')
plt.show()

# --------------------------
# --- 保存结果为 mat 文件 ---
# --------------------------
io.savemat(
    os.path.join(DATA_DIR, 'LinearDynamicSystem_Sol_Testset_DeepONet.mat'),
    {'X': X_test, 'U': U_test_pred}
)