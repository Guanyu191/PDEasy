import numpy as np
import torch


class _Dataset():
    def __init__(self, device=None, dtype=None):

        r"""Initialize the dataset object.

        This method is responsible for setting up the device and initializing the data dictionary.
        The device is selected based on whether CUDA is available. The data dictionary will be used to
        store various types of data that will be loaded and processed later.

        Args:
            None

        Attributes:
            device (torch.device): The device used to store and process data, preferring CUDA if available.
            data_dict (dict): An empty dictionary used to store different types of data in the dataset.

        Note:
            The `data_dict` is one of the cores of PDEasy. All data flow, including data generation
            and passing data into the model for training, occurs in the form of the `data_dict`.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

        self.data_dict = {}

    def array2tensor(self):
        for k, v in self.data_dict.items():
            if isinstance(v, np.ndarray):
                self.data_dict[k] = torch.from_numpy(v).float().to(self.device, self.dtype)
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

    def statistic(self, object_key="X_res", axis=0):
        self.data_dict["X_mean"] = self.data_dict[object_key].mean(axis=axis, keepdims=True)
        self.data_dict["X_std"] = self.data_dict[object_key].std(axis=axis, keepdims=True)

    def first_update(self, *args, **kwargs):
        self.custom_update(*args, **kwargs)
        self.statistic()
        self.array2tensor()

    def update(self, *args, **kwargs):
        self.tensor2array()
        self.custom_update(*args, **kwargs)
        self.array2tensor()

    def custom_update(self, *args, **kwargs):
        raise NotImplementedError

    def external_data(self):
        raise NotImplementedError