import numpy as np
import torch


class _Dataset():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dict = {}  # data dictionary 将数据存放于字典

    def array2tensor(self):
        for k, v in self.data_dict.items():
            if isinstance(v, np.ndarray):
                self.data_dict[k] = torch.from_numpy(v).float().to(self.device)
            else:
                actual_type = type(v).__name__
                raise ValueError(f"Current {k} is not a numpy array, but {actual_type}")
            
    def tensor2array(self):
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                self.data_dict[k] = v.detach().cpu().numpy()
            else:
                actual_type = type(v).__name__
                raise ValueError(f"Current {k} is not a numpy array, but {actual_type}")

    def statistic(self):
        self.data_dict["mean"] = self.data_dict["X_res"].mean(axis=0)
        self.data_dict["std"] = self.data_dict["X_res"].std(axis=0)

    def first_update(self, *args, **kwargs):
        self.custom_update(*args, **kwargs)              # 加载/更新所有数据

        self.statistic()                                 # 计算数据的统计信息，用作标准化
        self.array2tensor()                              # 将数据转到 cuda

    def update(self, *args, **kwargs):
        self.tensor2array()

        self.custom_update(*args, **kwargs)              # 加载/更新所有数据
        
        self.array2tensor()                              # 将数据转到 cuda

    def custom_update(self, *args, **kwargs):
        NotImplementedError

    def external_data(self):
        NotImplementedError