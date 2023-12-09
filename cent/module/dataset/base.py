"""This module implements the base class for datasets.
"""
from abc import ABC, abstractmethod

class BaseDatasetConfig(ABC):
    def __init__(self, env_config):
        self.env_config = env_config
        self.batch_size = 64
        self.usage = "train"

    @abstractmethod
    def dataset_root(self):
        return ""

class BaseDataset(ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement
    -- <__init__>: initialize the class; first call BaseDataset.__init__(self, opt).
    -- <__len__>: return the size of dataset.
    -- <__getitem__>: get a data point.
    """

    def __init__(self, config: BaseDatasetConfig):
        self.config = config
        self.name = "dummy"
    
    @abstractmethod 
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass 

    @property
    def batch_size(self):
        return self.config.batch_size

    def __str__(self):
        return self.name