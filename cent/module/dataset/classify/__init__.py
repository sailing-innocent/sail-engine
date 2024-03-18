# Dataset for classification problems

from module.config.env import BaseEnvConfig
from .fmnist.dataset import FMNIST, FMNISTConfig

def get_dataset_by_name(name: str, env_config: BaseEnvConfig):
    if name == "fmnist":
        config = FMNISTConfig(env_config)
        return FMNIST(config)

