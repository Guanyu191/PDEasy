'''
Descripttion: 
Author: Guanyu
Date: 2025-02-07 22:36:57
LastEditTime: 2025-02-08 13:52:58
FilePath: \PDEasy\dataset\rectangle.py
'''
import numpy as np

from dataset.dataset_base import _Dataset
from utils.sample_on_line import sample_on_line


class Dataset1D(_Dataset):
    def __init__(self, domain):
        super().__init__()
        self.x_min, self.x_max = domain

    def interior_random(self, n_res):
        """对内部随机采样"""
        x = np.random.rand(n_res) * (self.x_max - self.x_min) + self.x_min
        X_res = x.reshape(-1, 1)
        self.data_dict["X_res"] = X_res

    def interior_grid(self, n_x):
        """对内部网格采样"""
        x = np.linspace(self.x_min, self.x_max, n_x)
        X_res = x.reshape(-1, 1)
        self.data_dict["X_res"] = X_res

    def boundary(self):
        """对边界采样"""
        # 对于仅有 1D 空间的问题，边界条件只有两个端点
        b_min = np.array([self.x_min])
        b_max = np.array([self.x_max])
        X_bcs = np.stack([b_min, b_max], axis=0)

        self.data_dict["X_bcs"] = X_bcs


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
        self.data_dict["X_res"] = X_res

    def boundary_random(self, n_bcs):
        """对边界随机采样"""
        p_1 = (self.x_min, self.y_min)
        p_2 = (self.x_max, self.y_min)
        p_3 = (self.x_max, self.y_max)
        p_4 = (self.x_min, self.y_max)

        X_bcs = np.concatenate([
            sample_on_line(p_1, p_2, n_bcs, 'random'),
            sample_on_line(p_2, p_3, n_bcs, 'random'),
            sample_on_line(p_3, p_4, n_bcs, 'random'),
            sample_on_line(p_4, p_1, n_bcs, 'random')
        ], axis=0)
        self.data_dict["X_bcs"] = X_bcs

    def interior_grid(self, n_x, n_y):
        """对内部网格采样"""
        x = np.linspace(self.x_min, self.x_max, n_x)
        y = np.linspace(self.y_min, self.y_max, n_y)

        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        X_res = np.stack([x, y], axis=1)
        self.data_dict["X_res"] = X_res

    def boundary_grid(self, n_x, n_y):
        """对边界网格采样"""
        p_1 = (self.x_min, self.y_min)
        p_2 = (self.x_max, self.y_min)
        p_3 = (self.x_max, self.y_max)
        p_4 = (self.x_min, self.y_max)

        X_bcs = np.concatenate([
            sample_on_line(p_1, p_2, n_x, 'grid'),
            sample_on_line(p_2, p_3, n_y, 'grid'),
            sample_on_line(p_3, p_4, n_x, 'grid'),
            sample_on_line(p_4, p_1, n_y, 'grid')
        ], axis=0)
        self.data_dict["X_bcs"] = X_bcs


class Dataset2DT(_Dataset):
    def __init__(self, domain):
        super().__init__()
        self.x_min, self.x_max, self.y_min, self.y_max, self.t_min, self.t_max = domain

    def interior_random(self, n_res):
        """对内部随机采样"""
        x = np.random.rand(n_res) * (self.x_max - self.x_min) + self.x_min
        y = np.random.rand(n_res) * (self.y_max - self.y_min) + self.y_min
        t = np.random.rand(n_res) * (self.t_max - self.t_min) + self.t_min

        X_res = np.stack([x, y, t], axis=1)
        self.data_dict["X_res"] = X_res

    def boundary_random(self, n_bcs):
        """对边界随机采样"""
        x = np.repeat(self.x_min, n_bcs)
        y = np.random.rand(n_bcs) * (self.y_max - self.y_min) + self.y_min
        t = np.random.rand(n_bcs) * (self.t_max - self.t_min) + self.t_min
        b_x_min = np.stack([x, y, t], axis=1)

        x = np.repeat(self.x_max, n_bcs)
        b_x_max = np.stack([x, y, t], axis=1)

        x = np.random.rand(n_bcs) * (self.x_max - self.x_min) + self.x_min
        y = np.repeat(self.y_min, n_bcs)
        t = np.random.rand(n_bcs) * (self.t_max - self.t_min) + self.t_min
        b_y_min = np.stack([x, y, t], axis=1)

        y = np.repeat(self.y_max, n_bcs)
        b_y_max = np.stack([x, y, t], axis=1)

        X_bcs = np.concatenate([b_x_min, b_x_max, b_y_min, b_y_max], axis=0)
        self.data_dict["X_bcs"] = X_bcs

    def initial_random(self, n_ics):
        """对初始随机采样"""
        x = np.random.rand(n_ics) * (self.x_max - self.x_min) + self.x_min
        y = np.random.rand(n_ics) * (self.y_max - self.y_min) + self.y_min
        t = np.repeat(self.t_min, n_ics)

        X_ics = np.stack([x, y, t], axis=1)
        self.data_dict["X_ics"] = X_ics

    def interior_grid(self, n_x, n_y, n_t):
        """对内部网格采样"""
        x = np.linspace(self.x_min, self.x_max, n_x)
        y = np.linspace(self.y_min, self.y_max, n_y)
        t = np.linspace(self.t_min, self.t_max, n_t)

        x, y, t = np.meshgrid(x, y, t)


    # TODO
    def interior_grid(self, nx, ny, nt):
        """对内部网格采样"""
        pass

    # TODO
    def boundary_grid(self, nb):
        """对边界网格采样"""
        pass

    # TODO
    def initial_grid(self, n_x, n_y):
        """对初始网格采样"""
        pass
