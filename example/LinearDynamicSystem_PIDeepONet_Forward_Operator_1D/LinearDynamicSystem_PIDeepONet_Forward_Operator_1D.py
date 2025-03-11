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
from framework import PIDeepONetForward
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

N_BATCH = 10000
N_FUNC = 10000
N_SENSOR = 100
N_LOC = 10
N_DIM = 1

N_FUNC_TEST = N_FUNC
N_LOC_TEST = 10 * N_LOC

N_ITERS = 500
NN_BRANCH_LAYERS = [N_SENSOR] + [50]*5
NN_TRUNK_LAYERS = [N_DIM] + [50]*5


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
        F_res_train = trainset['F_res']  # (N_FUNC * N_LOC, 1)

        idx = np.random.choice(F_train.shape[0], N_BATCH, replace=False)
        self.data_dict['F'] = F_train[idx]
        self.data_dict['X'] = X_train[idx]
        self.data_dict['U'] = U_train[idx]
        self.data_dict['F_res'] = F_res_train[idx]
        self.data_dict['X_ics'] = np.zeros((N_BATCH, 1))

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
class DeepONet(PIDeepONetForward):
    def __init__(self, network_branch, network_trunk, should_normalize=True):
        super().__init__(network_branch, network_trunk, 
                         should_normalize=should_normalize)

    def forward(self, data_dict):
        # 读取 data_dict 的数据
        F, X, U, F_res, X_ics = [
            data_dict[key] for key in ['F', 'X', 'U', 'F_res', 'X_ics']
        ]
        
        loss_dict = {}
        loss_dict['pw_loss_operator'] = self.net_sol(F, X_ics) ** 2
        loss_dict['pw_loss_physics'] = self.net_res(F, X, F_res) ** 2

        return loss_dict
    
    def net_res(self, F, X, F_res):
        x = self.split_columns_and_requires_grad(X)
        u = self.net_sol(F, x)

        u_x = self.grad(u, x, 1)
        res = u_x - F_res
        return res


# ---------------------------------------------------
# --- 初始化训练实例 dataset pinn optimizer logger ---
# ---------------------------------------------------
dataset = Dataset(DOMAIN)

network_branch = MLP(NN_BRANCH_LAYERS, act_type='tanh')
network_trunk = MLP(NN_TRUNK_LAYERS, act_type='tanh')
deeponet = DeepONet(network_branch, network_trunk)
deeponet.F_mean, deeponet.F_std = dataset.data_dict['F_mean'], dataset.data_dict['F_std']
deeponet.X_mean, deeponet.X_std = dataset.data_dict['X_mean'], dataset.data_dict['X_std']
deeponet.U_mean, deeponet.U_std = dataset.data_dict['U_mean'], dataset.data_dict['U_std']

optimizer = optim.Adam(deeponet.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=1000)

log_keys = ['iter', 'loss', 'loss_operator', 'loss_physics', 'error_test']
logger = Logger(LOG_DIR, log_keys, num_iters=N_ITERS, print_interval=100)

def relative_l2_error(U_pred, U):
    return np.linalg.norm(U_pred - U) / np.linalg.norm(U)


# ---------------------------------
# --- 开始训练 打印并保存训练信息 ---
# ---------------------------------
best_loss = np.inf
for it in range(N_ITERS):
    deeponet.zero_grad()                                    # 清除梯度
    loss_dict = deeponet(dataset.data_dict)                 # 计算 point-wise loss
    
    pw_loss_operator = loss_dict["pw_loss_operator"]        # 提取 point-wise loss
    pw_loss_physics = loss_dict["pw_loss_physics"]

    loss_operator = torch.mean(pw_loss_operator)            # 计算 loss
    loss_physics = torch.mean(pw_loss_physics)
    loss = loss_operator + loss_physics

    loss.backward()                                         # 反向传播    
    optimizer.step()                                        # 更新网络参数
    scheduler.step(loss)                                    # 调整学习率

    with torch.no_grad():
        idx = np.random.choice(F_test.shape[0], 100, replace=False)
        U_test_pred = deeponet.net_sol(
            torch.from_numpy(F_test[idx]).float().to(deeponet.device), 
            torch.from_numpy(X_test[idx]).float().to(deeponet.device)
        ).detach().cpu().numpy()
        error_test = relative_l2_error(U_test_pred, U_test[idx])

    logger.record(                                          # 保存训练信息
        iter=it,                                            # 每隔一定次数自动打印
        loss=loss.item(),
        loss_operator=loss_operator.item(),
        loss_physics=loss_physics.item(),
        error_test=error_test,
    )
    
    if it % 1000 == 0:
        dataset.update()
                
    if loss.item() < best_loss:                             # 保存最优模型
        model_info = {
            'iter': it,
            'nn_state': deeponet.state_dict(),
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
deeponet.load_state_dict(model_info['nn_state'])
deeponet.eval()


# ----------------------------------
# --- 可视化 loss error solution ---
# ----------------------------------
plot_loss_from_logger(logger, FIGURE_DIR, show=True)
plot_error_from_logger(logger, FIGURE_DIR, show=True)

U_test_pred = deeponet.net_sol(
    torch.from_numpy(F_test).float().to(deeponet.device), 
    torch.from_numpy(X_test).float().to(deeponet.device)
).detach().cpu().numpy()
error = relative_l2_error(U_test_pred, U_test)

print('Relative l2 error: {:.3e}'.format(error))
with open(os.path.join(LOG_DIR, 'relative_error.txt'), 'w') as f_obj:
    f_obj.write('Relative l2 error: {:.3e}\n'.format(error))

N_PLOT = 3
X_test = X_test.reshape(N_FUNC_TEST, -1)
U_test = U_test.reshape(N_FUNC_TEST, -1)
U_test_pred = U_test_pred.reshape(N_FUNC_TEST, -1)
for _ in range(N_PLOT):
    idx = np.random.choice(U_test.shape[0], 1, replace=False).item()
    x = X_test[idx]
    sol = U_test[idx]
    sol_pred = U_test_pred[idx]

    plot_solution_from_data(
        FIGURE_DIR, 
        x_grid=x,
        sol=sol,
        sol_pred=sol_pred,

        x_label='$x$',
        y_label='$u$',

        x_ticks=np.linspace(0, 1, 5),
        y_ticks=np.linspace(-1, 1, 5),
        
        title_left=r'Solution $u(x)$',
        title_right=r'Absolute error',

        figure_name=f'Sol_PIDeepONet_Sample_{idx}.png'
    )


# --------------------------
# --- 保存结果为 mat 文件 ---
# --------------------------
io.savemat(
    os.path.join(DATA_DIR, 'LinearDynamicSystem_Sol_Testset_DeepONet.mat'),
    {'X': X_test, 'U': U_test_pred}
)