'''
Descripttion: 
Author: Guanyu
Date: 2025-02-07 22:36:57
LastEditTime: 2025-02-08 06:43:20
FilePath: \PDEasy\dataset\rectangle.py
'''
import numpy as np
import torch

from ..utils import sample_on_line


class _Dataset():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_dict = {}  # data dictionary 将数据存放于字典

    def array2tensor(self):
        for k, v in self.data_dict.items():
            self.data_dict[k] = torch.from_numpy(v).float().to(self.device)

    def tensor2array(self):
        for k, v in self.data_dict.items():
            self.data_dict[k] = v.detach().cpu().numpy()

    def statistic(self):
        self.data_dict["mean"] = self.data_dict["X_res"].mean(axis=0)
        self.data_dict["std"] = self.data_dict["X_res"].std(axis=0)


class Dataset1DT(_Dataset):
    def __init__(self, domain):
        super().__init__()
        self.x_min, self.x_max, self.t_min, self.t_max = domain

    def interior_random(self, n_res):
        """对内部随机采样"""
        x = np.random.rand(n_res) * (self.x_max - self.x_min) + self.x_min
        t = np.random.rand(n_res) * (self.t_max - self.t_min) + self.t_min

        X_res = np.stack([x, t], axis=1)
        self.data_dict["X_res"] = X_res

    def boundary_random(self, n_bcs):
        """对边界随机采样"""
        x = np.repeat(self.x_min, n_bcs)
        t = np.random.rand(n_bcs) * (self.t_max - self.t_min) + self.t_min
        b_min = np.stack([x, t], axis=1)

        x = np.repeat(self.x_max, n_bcs)
        b_max = np.stack([x, t], axis=1)

        X_bcs = np.concatenate([b_min, b_max], axis=0)
        self.data_dict["X_bcs"] = X_bcs

    def initial_random(self, n_ics):
        """对初始随机采样"""
        x = np.random.rand(n_ics) * (self.x_max - self.x_min) + self.x_min
        t = np.repeat(self.t_min, n_ics)

        X_ics = np.stack([x, t], axis=1)
        self.data_dict["X_ics"] = X_ics

    def interior_grid(self, n_x, n_t):
        """对内部网格采样"""
        x = np.linspace(self.x_min, self.x_max, n_x)
        t = np.linspace(self.t_min, self.t_max, n_t)

        x, t = np.meshgrid(x, t)
        x, t = x.flatten(), t.flatten()
        X_res = np.stack([x, t], axis=1)
        self.data_dict["X_res"] = X_res

    def boundary_grid(self, n_x):
        """对边界网格采样"""
        x = np.repeat(self.x_min, n_x)
        t = np.linspace(self.t_min, self.t_max, n_x)
        b_min = np.stack([x, t], axis=1)

        x = np.repeat(self.x_max, n_x)
        b_max = np.stack([x, t], axis=1)

        X_bcs = np.concatenate([b_min, b_max], axis=0)
        self.data_dict["X_bcs"] = X_bcs

    def initial_grid(self, n_x):
        """对初始网格采样"""
        x = np.linspace(self.x_min, self.x_max, n_x)
        t = np.repeat(self.t_min, n_x)

        X_ics = np.stack([x, t], axis=1)
        self.data_dict["X_ics"] = X_ics


class Dataset2D(_Dataset):
    def __init__(self, domain):
        super().__init__()
        self.x_min, self.x_max, self.y_min, self.y_max = domain

    def interior_random(self, n_res):
        """对内部随机采样"""
        x = np.random.rand(n_res) * (self.x_max - self.x_min) + self.x_min
        y = np.random.rand(n_res) * (self.y_max - self.y_min) + self.y_min

        X_res = np.stack([x, y], axis=1)
        

    

    
    


    def interior_random(self, n_res):
        """对内部随机采样"""
        x = np.random.rand(n_res) * (self.x_max - self.x_min) + self.x_min
        y = np.random.rand(n_res) * (self.y_max - self.x_min) + self.x_min
        X_res = np.stack([x, y], axis=1)
        self.data_dict["X_res"] = X_res

    def boundary_random(self, n_bcs):
        """对边界网格采样"""
        b1 = self.sample_on_line(self.x_min, self.y_min, self.x_max, self.y_min, n_bcs)
        b2 = self.sample_on_line(self.x_min, self.y_max, self.x_max, self.y_max, n_bcs)
        b3 = self.sample_on_line(self.x_min, self.y_min, self.x_min, self.y_max, n_bcs)
        b4 = self.sample_on_line(self.x_max, self.y_min, self.x_max, self.y_max, n_bcs)
        X_bcs = np.concatenate([b1, b2, b3, b4], axis=0)
        self.data_dict["X_bcs"] = X_bcs