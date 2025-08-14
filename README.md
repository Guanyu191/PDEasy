# PDEasy ​(​0​.​1​.​3)​ :zap:

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**PDEasy: Lightweight PINN & Operator PDE Solver for Research, Balancing Abstraction and Flexibility for Algorithm Innovation.**

> **Note:** Currently, most Python libraries for Physics-Informed Neural Networks (PINN) and Operator Learning are highly encapsulated, mainly catering to engineering deployment or beginners. However, for researchers working on PINN-related projects, it's extremely difficult to implement their own ideas using highly encapsulated code. Therefore, we aim to develop a PINN-PDE solver library tailored for researchers, which strikes a balance between encapsulation and extensibility. This will enable users to quickly implement new ideas and expedite algorithm innovation. 

> **Log:**
>
> - 0.1.0  |  Implemented a framework for solving forward and inverse problems using PINN, supporting multiple inputs and outputs, as well as convenient differentiation.
> - 0.1.1  |  Implemented the example of 1D problems using Deep Operator Network (DeepONet). Refactored `pinn` into `framework` to be compatible with the operator learning framework.

> - 0.1.2  |  Implemented the example of 1D problems using Physics-informed DeepONet (PIDeepONet) and added examples for high - frequency problems.
> - 0.1.3  |  Packaged the library as a Python package that can be installed and invoked via `pip`. Fixed several bugs.
> - 0.1.4  |  Implemented the example of 2D Poisson using Elementary Learning Machine (ELM). Optimized some bugs.

## :package: Installation

- Step 1: Clone pdeasy

  `git clone https://github.com/Guanyu191/PDEasy.git`

- Step 2: Enter the root directory of pdeasy and run

  `pip install .`

## :rocket: Reference

> 1. [Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational physics, 2019, 378: 686-707.](https://www.sciencedirect.com/science/article/am/pii/S0021999118307125)
> 2. [Rasht‐Behesht M, Huber C, Shukla K, et al. Physics‐informed neural networks (PINNs) for wave propagation and full waveform inversions[J]. Journal of Geophysical Research: Solid Earth, 2022, 127(5): e2021JB023120.](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021JB023120)
> 3. [Zhang G, Duan Y, Pan G, et al. Data-driven discovery of state-changes in underlying system from hidden change-points in partial differential equations with spatiotemporal varying coefficients[J]. Journal of Computational and Applied Mathematics, 2025: 116962.](https://www.sciencedirect.com/science/article/pii/S0377042725004765)
> 4. [Lu L, Jin P, Pang G, et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators[J]. Nature machine intelligence, 2021, 3(3): 218-229.](https://www.nature.com/articles/s42256-021-00302-5)
> 5. [Wang S, Wang H, Perdikaris P. Learning the solution operator of parametric partial differential equations with physics-informed DeepONets[J]. Science advances, 2021, 7(40): eabi8605.](https://www.science.org/doi/abs/10.1126/sciadv.abi8605)
> 6. [Wang S, Wang H, Perdikaris P. On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks[J]. Computer Methods in Applied Mechanics and Engineering, 2021, 384: 113938.](https://www.sciencedirect.com/science/article/pii/S0045782521002759)
> 7. [Quan H D, Huynh H T. Solving partial differential equation based on extreme learning machine[J]. Mathematics and Computers in Simulation, 2023, 205: 697-708.](https://www.sciencedirect.com/science/article/pii/S0378475422004323)
> 8. ...

## 🔑 Key Features

### 1. Streamlined PINN Solving Process

- **PINN Solving Process**: The process is standardized as follows: defining hyperparameters, defining the dataset, defining the neural network model, defining the PINN model, training the model, and evaluating and visualizing the results.

- Among these steps, various **neural network** models, **PINN forward and inverse problem** models, and visualization modules are all encapsulated, and extension interfaces are provided.
- In particular, the process of calculating **partial derivatives** of input coordinates, which is unique to PINN, has been simplified.

### 2. Flexible Implementation of Custom Ideas

- Data within the framework, such as **sampling points** and **loss** information, flows in the form of **dictionaries** (Python `dict`). Users can easily read or add information and integrate algorithms related to sampling and weighting.
- The training module is not fully encapsulated, allowing users to modify the training process more easily.
- The PINN model provides excellent extension interfaces, enabling the integration of various neural networks and the addition of different calculation methods.

### 3. Comprehensive Monitoring of Training Information

- During the training process, **loss** and **error** can be quickly calculated through the encapsulated modules and stored in the **logger**.
- The visualization module can directly plot graphs using the logger to display the changes in various losses and errors during the training process.

> **Note:** PDEasy is built on top of popular Python packages such as PyTorch, NumPy, and Matplotlib.

## :hourglass_flowing_sand: Quick Start
Here, we take the **forward problem of the Burgers' equation** as an example to introduce the usage of PDEasy. The complete code can be found at `./example/Burgers_Forward_1DT/Burgers_Forward_1DT.py`.

> **Note:** In the `example` folder, we have implemented a total of 7 examples in 1D and 2D spaces (along with time), covering both forward and inverse problems. These examples incorporate various types of neural networks such as MLP, ResNet, Fourier Feature Network, and Multi - Head Network. They also demonstrate the integration of algorithms like boundary constraints, scaling, and adaptive loss weighting. More examples will be made public in the future.

### 0. Import PDEasy module

```python
from pdeasy.dataset import Dataset1DT
from pdeasy.framework import PINNForward
from pdeasy.network import MLP
from pdeasy.utils import *
from pdeasy.plotting import *
```

### 1. Define hyperparameters

```python
DATA_DIR = './data'
FIGURE_DIR = './figure'
LOG_DIR = './log'
MODEL_DIR = './model'

DOMAIN = (-1, 1, 0, 1)			# (x_min, x_max, t_min, t_max)
N_RES = 2000					# number of residual points
N_BCS = 200						# number of boundary points
N_ICS = 200						# number of initial points
N_ITERS = 20000					# number of iterations for training
NN_LAYERS = [2] + [40]*4 + [1]	# network architectures
```

### 2. Define the dataset

```python
class Dataset(Dataset1DT):
    def __init__(self, domain):
        super().__init__(domain)

    def custom_update(self, n_res=N_RES, n_bcs=N_BCS, n_ics=N_ICS):
        self.interior_random(n_res)		# sampling points
        self.boundary_random(n_bcs)
        self.initial_random(n_ics)
```

### 3. Define the PINN model

```python
class PINN(PINNForward):
    def __init__(self, network_solution, should_normalize=True):
        super().__init__(network_solution, should_normalize)

    def forward(self, data_dict):
        # read data from data_dict
        X_res, X_bcs, X_ics = data_dict["X_res"], data_dict["X_bcs"], data_dict["X_ics"]

        # culculate point-wise loss
        # it facilitates for weight strategies
        loss_dict = {}
        loss_dict['pw_loss_res'] = self.net_res(X_res) ** 2
        loss_dict['pw_loss_bcs'] = self.net_bcs(X_bcs) ** 2
        loss_dict['pw_loss_ics'] = self.net_ics(X_ics) ** 2

        return loss_dict
    
    def net_res(self, X):
        x, t = self.split_columns_and_requires_grad(X)	# extract the inputs in each columns
        u = self.net_sol([x, t])						# get the solution

        u_x = self.grad(u, x, 1)						# culculate the derivative
        u_t = self.grad(u, t, 1)
        u_xx = self.grad(u, x, 2)
        res_pred = u_t + u * u_x - (0.01 / torch.pi) * u_xx
        return res_pred
    
    def net_bcs(self, X):
        u = self.net_sol(X)
        bcs_pred = u - 0
        return bcs_pred
    
    def net_ics(self, X):
        u = self.net_sol(X)
        ics_pred = u + torch.sin(torch.pi * X[:, [0]])
        return ics_pred
```

### 4. Training

#### 4.1 Initialize the training instance

```python
dataset = Dataset(DOMAIN)		# generate an instance of the dataset

network = MLP(NN_LAYERS)		# generate an instance of the network
pinn = PINN(network)			# generate an instance of the pinn model
pinn.X_mean, pinn.X_std = dataset.data_dict['X_mean'], dataset.data_dict['X_std']

optimizer = optim.Adam(pinn.parameters(), lr=0.001)

log_keys = ['iter', 'loss', 'loss_res', 'loss_bcs', 'loss_ics', 'error_u']
logger = Logger(LOG_DIR, log_keys, num_iters=N_ITERS, print_interval=100)

```

#### 4.2 Training loop

```python
best_loss = np.inf
for it in range(N_ITERS):
    pinn.zero_grad()						# clear the gradients
    loss_dict = pinn(dataset.data_dict)		# culculate point-wise loss
    
    pw_loss_res = loss_dict["pw_loss_res"]	# extact point-wise loss
    pw_loss_bcs = loss_dict["pw_loss_bcs"]
    pw_loss_ics = loss_dict["pw_loss_ics"]
    
    loss_res = torch.mean(pw_loss_res)		# culculate loss
    loss_bcs = torch.mean(pw_loss_bcs)		# easily add some algorithm
    loss_ics = torch.mean(pw_loss_ics)
        
    loss = loss_res + loss_bcs + loss_ics
    
    loss.backward()							# backpropagation  
    optimizer.step()						# update the network parameters

    error_u, _ = relative_error_of_solution(pinn, ref_data=(X, u), num_sample=500)

    logger.record(							# save the training information
        iter=it,							# automatically print
        loss=loss.item(),
        loss_res=loss_res.item(),
        loss_bcs=loss_bcs.item(),
        loss_ics=loss_ics.item(),
        error_u=error_u
    )
    
    if it % 100 == 0:
        dataset.update()
    
    if loss.item() < best_loss:				# save the best model
        model_info = {
            'iter': it,
            'nn_state': pinn.state_dict(),
        }
        torch.save(model_info, os.path.join(MODEL_DIR, 'model.pth'))
        best_loss = loss.item()

logger.print_elapsed_time()
logger.save()
```

### 5. Evaluate and visualize

#### 5.1 Import the training information and model parameters

```python
logger.load()

model_info = torch.load(os.path.join(MODEL_DIR, 'model.pth'))
pinn.load_state_dict(model_info['nn_state'])
pinn.eval()
```

#### 5.2 Visualize the loss and error

```python
plot_loss_from_logger(logger, FIGURE_DIR, show=True)
plot_error_from_logger(logger, FIGURE_DIR, show=True)
```

#### 5.3 Visualize the solution

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
We welcome more developers to join us. Please contact us through the following channels:

- Email: guanyu191@163.com
- WeChat: guanyu191

> **Note:** The predecessor of PDEasy was developed by Guanyu Pan and Zikun Xu from the DataHub team in January 2023 according to the team's scientific research needs. At that time, it was mainly used for solving inverse problems with the Physics-Informed Neural Network (PINN). In February 2025, we redesigned and improved the code, and developed the PDEasy library, which is targeted at scientific researchers in the field of PINN and Operator Learning. In the future, we will further improve it to support more forward and inverse problems of Partial Differential Equations (PDEs).

## :scroll: License
MIT License. See [LICENSE](LICENSE) for details.

## :star: Citation
If using PDEasy in research:
```bibtex
@software{PDEasy,
  author = {Guanyu Pan},
  title = {PDEasy: Lightweight PINN & Operator PDE Solver for Research, Balancing Abstraction and Flexibility for Algorithm Innovation.},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Guanyu191/PDEasy}},
}
```
