# from http://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html
from ..base import RegressionDataset, RegressionDatasetConfig, data_iter
import matplotlib.pyplot as plt
import torch 

def synthetic_linear_dataset(w, b, N):
    X = torch.normal(0, 1, (N, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

class LinearDatasetConfig(RegressionDatasetConfig):
    """
    This class implements the configuration for the linear dataset.
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        self.batch_size = 10
        self.input_dim = 2
        self.output_dim = 1
        self.true_w = torch.tensor([2, -3.4], dtype=torch.float32)
        self.true_b = torch.tensor([4.2], dtype=torch.float32)
    
    def dataset_root(self):
        return ""  

class LinearDataset(RegressionDataset):
    """
    This class implements the linear dataset.
    """
    def __init__(self, config: LinearDatasetConfig):
        super().__init__(config)
        self.name = "linear"
        self._features, self._labels = synthetic_linear_dataset(config.true_w, config.true_b, config.sample_size)
    
    def features(self):
        return self._features
    
    def labels(self):
        return self._labels

    def visualize(self, axis=1):
        # the scatter for the second feature and label
        x = self._features[:, (axis)].detach().numpy()
        y = self._labels.detach().numpy()
        plt.scatter(x, y)
        plt.show()

    def __iter__(self):
        return iter(data_iter(
            self.config.batch_size, self._features, self._labels))

    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, index):
        return self._features[index], self._labels[index]
