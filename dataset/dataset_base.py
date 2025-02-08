import torch


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

    def update_dataset(self):
        NotImplementedError

    def external_data(self):
        NotImplementedError