from ..base import RegressionDataset, RegressionDatasetConfig, data_iter
from module.config.env import BaseEnvConfig 

import torch 

class XSNIXDataset1DConfig(RegressionDatasetConfig):
    def __init__(self, env_config: BaseEnvConfig):
        super().__init__(env_config)
        self.sample_size = 1000
        self.input_dim = 1
        self.output_dim = 1
        self.input_range: tuple = (1, 10) 
        self.batch_size = 10
    
class XSNIXDataset1D(RegressionDataset):
    def __init__(self, config: XSNIXDataset1DConfig):
        super().__init__(config)
        self.name = "xsinx dataset"
        x = torch.linspace(
            config.input_range[0], 
            config.input_range[1], config.sample_size)
        y = x * torch.sin(x)
        y = y + torch.normal(0, 0.1, y.shape)
        self._features = x.reshape(-1, 1)
        self._labels = y.reshape(-1, 1)

    def features(self):
        return self._features
    
    def labels(self):
        return self._labels

    def __iter__(self):
        return iter(data_iter(
            self.config.batch_size, self._features, self._labels))

    def __len__(self):
        return len(self._features)
    
    def __getitem__(self, idx): 
        return self._features[idx], self._labels[idx]

    def visualize(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.features(), self.labels())
        plt.show()