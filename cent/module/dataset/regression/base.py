from module.dataset.base import BaseDataset, BaseDatasetConfig 
from abc import ABC, abstractmethod 
from module.config.env import BaseEnvConfig 

import random
import torch 

def data_iter(batch_size, features, labels):
    N = len(features)
    indices = list(range(N))
    random.shuffle(indices)
    for i in range(0, N, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, N)])
        yield features[batch_indices], labels[batch_indices]

class RegressionDatasetConfig(BaseDatasetConfig):
    def __init__(self, env_config: BaseEnvConfig):
        super().__init__(env_config)
        self.input_dim = 2
        self.output_dim = 1
        self.sample_size = 1000

    def dataset_root(self):
        return ""

class RegressionDataset(BaseDataset, ABC):
    def __init__(self, config: RegressionDatasetConfig):
        super().__init__(config)

    @abstractmethod
    def features(self):
        return []

    @abstractmethod
    def labels(self):
        return []

    def __len__(self):
        return 0

    def __getitem__(self, index):
        pass