import os
import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim

from pdeasy.dataset import Dataset2D
from pdeasy.framework import PINNForward
from pdeasy.network import *
from pdeasy.utils import *
from pdeasy.plotting import *


# -----------------------
# --- Hyperparameters ---
# -----------------------
DATA_DIR = './data'
FIGURE_DIR = './figure'
LOG_DIR = './log'
MODEL_DIR = './model'

DOMAIN = (-1, 1, -1, 1)  # (x_min, x_max)
N_X = 101
N_Y = 101
N_BCS = 2000
N_ITERS = 1
NN_LAYERS = [2] + [256]*1 + [1]

DTYPE = torch.float64


# ---------------------------------------------------------------------
# --- Load reference solution for culculating the relative L2 error ---
# ---------------------------------------------------------------------
init_dir(DATA_DIR, FIGURE_DIR, LOG_DIR, MODEL_DIR)

data = io.loadmat(os.path.join(DATA_DIR, 'Poisson_Sol.mat'))
u = data['u']  # shape (N_y, N_x)
x = data['x'].flatten()
y = data['y'].flatten()

u_shape = u.shape
xx, yy = np.meshgrid(x, y)
xx, yy, u = xx.flatten(), yy.flatten(), u.flatten()

X = np.stack([xx, yy], axis=1)


# --------------------------
# --- Customized dataset ---
# --------------------------
class Dataset(Dataset2D):
    def __init__(self, domain):
        super().__init__(domain, dtype=DTYPE)

    def custom_update(self, n_x=N_X, n_y=N_Y, n_bcs=N_BCS):
        self.interior_grid(n_x, n_y)
        self.boundary_random(n_bcs)


# ---------------
# --- ELM Net ---
# ---------------
class ELMNet(nn.Module):
    def __init__(self, network_solution: nn.Module, lmbd: float = 1e-6):
        super().__init__()

        self.network_solution = network_solution
        for p in network_solution.parameters():
            p.requires_grad_(False)  # freeze the network

        # split the network into two parts: 'out' and 'before_out'
        self.out = [m for m in network_solution.modules() if isinstance(m, nn.Linear)][-1]
        self.before_out = nn.Sequential(*list(network_solution.model.children())[:-1])
        self.lmbd = lmbd  # regularization parameter
        self.to('cuda' if torch.cuda.is_available() else 'cpu', DTYPE)

    def forward(self, data_dict):
        X_res, X_bcs = data_dict['X_res'], data_dict['X_bcs']

        r"""
        Construct a linear system: H \beta = K
            H = [H_res, H_bcs]
            K = [K_res, K_bcs]
        Then solve with (H^T H + \lambda I) \beta = H^T K
        reference: https://doi.org/10.1016/j.matcom.2022.10.018
        """
        N_RES, N_DIM = X_res.shape
        H_res = self.net_res_elm(X_res)
        H_bcs = self.net_bcs_elm(X_bcs)
        H = torch.cat([H_res, H_bcs], dim=0)

        alpha, x0, y0 = -10.0, 0.5, 0.5
        def f(x, y):
            
            r2 = (x - x0)**2 + (y - y0)**2
            rhs = 4 * alpha * (1 + alpha * r2) * np.exp(alpha * r2)
            return rhs
        def u(x, y):
            return torch.exp(alpha * ((x - x0)**2 + (y - y0)**2))

        K_res = f(X_res[:, [0]], X_res[:, [1]])
        K_bcs = u(X_bcs[:, [0]], X_bcs[:, [1]])
        K = torch.cat([K_res, K_bcs], dim=0)

        # solve the linear system
        I = torch.eye(H.size(1), device=H.device)
        beta = torch.linalg.solve(H.T @ H + self.lmbd * I, H.T @ K)

        self.out.weight.data = beta.T

        pw_loss = (H @ beta - K) ** 2

        loss_dict = {}
        loss_dict['pw_loss_res'] = pw_loss[:N_RES]
        loss_dict['pw_loss_bcs'] = pw_loss[N_RES:N_RES+N_BCS*4]
        return loss_dict
    
    @torch.no_grad()
    def net_res_elm(self, X, h=1e-3):
        """Center difference"""
        N, D = X.shape
        h = torch.as_tensor(h, dtype=X.dtype, device=X.device)

        x, y = X[:, [0]], X[:, [1]]
        X_px = torch.cat((x + h, y), 1)  # (x+h, y)
        X_mx = torch.cat((x - h, y), 1)  # (x-h, y)
        X_py = torch.cat((x, y + h), 1)  # (x, y+h)
        X_my = torch.cat((x, y - h), 1)  # (x, y-h)

        X = torch.cat((X, X_px, X_mx, X_py, X_my), 0)
        H = self.before_out(X)
        H, H_px, H_mx, H_py, H_my = H.split(N)

        inv_h2 = 1.0 / (h * h)
        H_xx = (H_px - 2.0 * H + H_mx) * inv_h2
        H_yy = (H_py - 2.0 * H + H_my) * inv_h2
        return H_xx + H_yy

    @torch.no_grad()
    def net_bcs_elm(self, X):
        return self.before_out(X)
        
    def net_sol(self, X):
        return self.network_solution(X)
    

# ------------------------------------------------
# --- Initialize dataset ELM optimizer logger ---
# ------------------------------------------------
dataset = Dataset(DOMAIN)

# two ways to initilize network
# the brief one is: 
# network = MLP(NN_LAYERS, init_type='uniform', a=-1, b=1)
# the direct one is: 
network = MLP(NN_LAYERS)
for m in network.model.modules():
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-1, b=1)

elm = ELMNet(network)


# ----------------
# --- Training ---
# ----------------
loss_dict = elm(dataset.data_dict)

pw_loss_res = loss_dict["pw_loss_res"]
pw_loss_bcs = loss_dict["pw_loss_bcs"]
loss_res = torch.mean(pw_loss_res)
loss_bcs = torch.mean(pw_loss_bcs)
loss = loss_res + loss_bcs

error_u, _ = relative_error_of_solution(elm, ref_data=(X, u), num_sample=1024)

model_info = {
    'iter': 1,
    'nn_state': elm.state_dict(),
}
torch.save(model_info, os.path.join(MODEL_DIR, 'model.pth'))


# ------------------
# --- Load model ---
# ------------------
model_info = torch.load(os.path.join(MODEL_DIR, 'model.pth'), map_location=torch.device('cpu'))
elm.load_state_dict(model_info['nn_state'])
elm.eval()


# -------------------------------------
# --- Visualize loss error solution ---
# -------------------------------------
error_u, u_pred = relative_error_of_solution(elm, ref_data=(X, u))

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


# ---------------------------------
# --- Save ELM solution as mat ---
# ---------------------------------
io.savemat(
    os.path.join(DATA_DIR, 'Poisson_Sol_ELM.mat'),
    {'x': x, 'u': u_pred}
)