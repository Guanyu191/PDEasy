# PDEasy ​(​0​.​1​.​2)​ :zap:

[![PyPI Version](https://img.shields.io/pypi/v/pdeasy)](https://pypi.org/project/pdeasy/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**PDEasy: Lightweight PINN & Operator PDE Solver for Research, Balancing Abstraction and Flexibility for Algorithm Innovation.**

> **Note:** 目前, 关于 PINN (Physics-Informed Neural Networks) 的 Python 库大都封装程度较高, 主要面向工程部署或新手入门. 然而, 对于 PINN 相关科研工作者来说, 通过较高封装的代码去实现自己的 idea 是非常困难的. 因此, 我们希望写一套面向科研工作者的 PINN-PDE 求解库, 平衡封装程度与扩展性, 使得用户能够快速实现新的 idea, 加速算法创新.

> **Log:**
>
> - 0.1.0，实现了 PINN 求解正反问题的框架，支持多输入输出、便捷求导等.
> - 0.1.1，实现了 DeepONet 求解 1D 算例，将 `pinn` 重构为 `framework`，兼容算子学习框架.
> - 0.1.2，实现了 Physics-informed DeepONet 求解 1D 算例，增加了高频问题的算例.

## :rocket: Reference

> 1. ...
> 2. ...

## 🔑 Key Features

### 1. PINN 求解流程化

- **PINN 求解流程**规范为: 定义超参数, 定义数据集, 定义网络模型, 定义 PINN 模型, 训练模型, 评估与可视化.

- 其中, 各种**网络**模型, **PINN 正反问题**模型, 可视化模块等都实现了封装, 并提供了扩展接口.
- 尤其是对于 PINN 特有的对输入坐标**求偏导**, 也实现了简化.

### 2. 可以灵活扩展自己的 idea

- 框架内的数据, 例如**采样点**和 **loss** 信息, 都**以 dict 的形式流动**, 用户可以轻松地读取或加入信息, 并融合采样和权重等算法.
- 未封装训练模块, 使用户更容易修改训练过程.
- PINN 模型提供了良好的扩展接口, 可以融合各种网络, 或增加各种计算方式.

### 3. 全面的训练信息监控

- 训练过程中的 **loss** 和 **error**, 都可以通过封装模块快速计算, 并存储到 **logger** 中.
- 可视化模块可以直接通过 logger 绘图, 展示训练过程的各项 loss 和 error 的变化.

> **Note:** PDEasy 基于 PyTorch, NumPy, Matplotlib 等主流 Python packages.

## :package: Installation

目前仅支持下载压缩包使用.

## :hourglass_flowing_sand: Quick Start
这里我们以 Burgers 方程正问题为例, 介绍 PDEasy 的使用方法. 完整代码见 `./example/Burgers_Forward_1DT/Burgers_Forward_1DT.py`.

> **Note:** 在 `example`文件夹中, 我们实现了 1D, 2D 空间 (和时间)的算例, 包括正问题和反问题, 共有 7 个. 其中, 结合了 MLP, ResNet, Fourier Feature Network, Multi-Head Network 等各种网络类型, 以及展示了边界约束, 尺度放缩, 自适应 loss 权重等算法的融合, 后续会公开更多的算例.

### 0. 导入库

```python
import sys
sys.path.append("../../")

from dataset import Dataset1DT
from framework import PINNForward
from network import MLP
from utils import *
from plotting import *
```

### 1. 定义超参数

```python
DATA_DIR = './data'
FIGURE_DIR = './figure'
LOG_DIR = './log'
MODEL_DIR = './model'

DOMAIN = (-1, 1, 0, 1)  			# (x_min, x_max, t_min, t_max)
N_RES = 2000    					# 内部点数量
N_BCS = 200	    					# 边界点数量
N_ICS = 200	    					# 初始点数量
N_ITERS = 20000    					# 网络训练迭代次数
NN_LAYERS = [2] + [40]*4 + [1]		# 网络层
```

### 2. 定义数据集

```python
class Dataset(Dataset1DT):
    def __init__(self, domain):
        super().__init__(domain)

    def custom_update(self, n_res=N_RES, n_bcs=N_BCS, n_ics=N_ICS):
        self.interior_random(n_res)    # 在内部随机采样 n_res 个点
        self.boundary_random(n_bcs)    # 在边界随机采样 n_res 个点
        self.initial_random(n_ics)     # 在初始时刻随机采样 n_res 个点
```

### 3. 定义 PINN 模型

```python
class PINN(PINNForward):
    def __init__(self, network_solution, should_normalize=True):
        super().__init__(network_solution, should_normalize)

    def forward(self, data_dict):
        # 读取 data_dict 的数据
        X_res, X_bcs, X_ics = data_dict["X_res"], data_dict["X_bcs"], data_dict["X_ics"]

        # 计算 point-wise loss
        # 便于后续引入权重策略
        loss_dict = {}
        loss_dict['pw_loss_res'] = self.net_res(X_res) ** 2
        loss_dict['pw_loss_bcs'] = self.net_bcs(X_bcs) ** 2
        loss_dict['pw_loss_ics'] = self.net_ics(X_ics) ** 2

        return loss_dict
    
    def net_res(self, X):
        x, t = self.split_columns_and_requires_grad(X)			# 将多列的输入分别提取出来
        u = self.net_sol([x, t])								# 获得输出解

        u_x = self.grad(u, x, 1)								# 求 u 对 x 的 1 阶导
        u_t = self.grad(u, t, 1)								# 求 u 对 t 的 1 阶导
        u_xx = self.grad(u, x, 2)								# 求 u 对 x 的 2 阶导
        res_pred = u_t + u * u_x - (0.01 / torch.pi) * u_xx		# 构造方程残差
        return res_pred
    
    def net_bcs(self, X):
        u = self.net_sol(X)										# 获得输出解
        bcs_pred = u - 0										# 构造边界条件残差
        return bcs_pred
    
    def net_ics(self, X):
        u = self.net_sol(X)										# 获得输出解
        ics_pred = u + torch.sin(torch.pi * X[:, [0]])			# 构造初始条件残差
        return ics_pred
```

### 4. 训练模型

#### 4.1 初始化训练实例

```python
dataset = Dataset(DOMAIN)		# 生成数据集实例

network = MLP(NN_LAYERS)		# 生成网络实例
pinn = PINN(network)			# 生成 PINN 实例
pinn.X_mean, pinn.X_std = dataset.data_dict['X_mean'], dataset.data_dict['X_std']

optimizer = optim.Adam(pinn.parameters(), lr=0.001)

log_keys = ['iter', 'loss', 'loss_res', 'loss_bcs', 'loss_ics', 'error_u']
logger = Logger(LOG_DIR, log_keys, num_iters=N_ITERS, print_interval=100)

```

#### 4.2 训练

```python
best_loss = np.inf
for it in range(N_ITERS):
    pinn.zero_grad()                                        # 清除梯度
    loss_dict = pinn(dataset.data_dict)                     # 计算 point-wise loss
    
    pw_loss_res = loss_dict["pw_loss_res"]                  # 提取 point-wise loss
    pw_loss_bcs = loss_dict["pw_loss_bcs"]
    pw_loss_ics = loss_dict["pw_loss_ics"]
    
    loss_res = torch.mean(pw_loss_res)                      # 计算 loss
    loss_bcs = torch.mean(pw_loss_bcs)						# 可以引入权重算法
    loss_ics = torch.mean(pw_loss_ics)
        
    loss = loss_res + loss_bcs + loss_ics
    
    loss.backward()                                         # 反向传播    
    optimizer.step()                                        # 更新网络参数

    error_u, _ = relative_error_of_solution(pinn, ref_data=(X, u), num_sample=500)

    logger.record(                                          # 保存训练信息
        iter=it,                                            # 每隔一定次数自动打印
        loss=loss.item(),
        loss_res=loss_res.item(),
        loss_bcs=loss_bcs.item(),
        loss_ics=loss_ics.item(),
        error_u=error_u
    )
    
    if it % 100 == 0:
        dataset.update()
    
    if loss.item() < best_loss:                             # 保存最优模型
        model_info = {
            'iter': it,
            'nn_state': pinn.state_dict(),
        }
        torch.save(model_info, os.path.join(MODEL_DIR, 'model.pth'))
        best_loss = loss.item()

logger.print_elapsed_time()
logger.save()
```

### 5. 评估与可视化

#### 5.1 导入训练信息以及模型参数

```python
logger.load()

model_info = torch.load(os.path.join(MODEL_DIR, 'model.pth'))
pinn.load_state_dict(model_info['nn_state'])
pinn.eval()
```

#### 5.2 可视化 loss 和 error

```python
plot_loss_from_logger(logger, FIGURE_DIR, show=True)
plot_error_from_logger(logger, FIGURE_DIR, show=True)
```

#### 5.3 可视化 solution

```python
error_u, u_pred = relative_error_of_solution(pinn, ref_data=(X, u))

plot_solution_from_data(
    FIGURE_DIR, 
    x_grid=xx.reshape(u_shape),
    y_grid=tt.reshape(u_shape),
    sol=u.reshape(u_shape),
    sol_pred=u_pred.reshape(u_shape),

    x_label='$x$',
    y_label='$t$',

    x_ticks=np.linspace(-1, 1, 5),
    y_ticks=np.linspace(0, 1, 5),
    
    title_left=r'Reference $u(x,t)$',
    title_middle=r'Predicted $u(x,t)$',
    title_right=r'Absolute error'
)
```

## :books: Documentation
TODO.

## :handshake: Contributing
我们欢迎更多开发者加入, 请联系:

- 邮箱 guanyu191@163.com
- 微信 guanyu191

> **Note:** PDEasy 前身由数据谷团队的潘冠宇和徐梓锟在 2023 年 1 月根据团队科研需求开发, 当时主要用于 PINN 求解反问题. 在 2025 年 2 月, 我们重新改进代码, 面向 PINN 领域科研工作者, 开发了 PDEasy 库. 往后我们会进一步完善, 支持更多的 PDE 正反问题.

## :scroll: License
MIT License. See [LICENSE](LICENSE) for details.

## :star: Citation
If using PDEasy in research:
```bibtex
@software{PDEasy,
  author = {Guanyu Pan},
  title = {PDEasy: Lightweight PINN-PDE Solver for Research, Balancing Abstraction and Flexibility for Algorithm Innovation.},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Guanyu191/PDEasy}},
}
```
