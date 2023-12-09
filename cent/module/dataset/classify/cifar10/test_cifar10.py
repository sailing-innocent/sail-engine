import pytest 

from .dataset import CIFAR10, CIFAR10Config
from module.config.env import get_env_config

@pytest.mark.current 
def test_cifar10():
    env_config = get_env_config()
    config = CIFAR10Config(env_config)
    dataset = CIFAR10(config)

    assert len(dataset) == 50000
    assert dataset.batch_size == 64
    sample, label = dataset[0]
    assert sample.shape == (3, 32, 32)
    assert label == 6
    for sample, label in dataset:
        assert sample.shape == (64, 3, 32, 32)
        assert label.shape == (64,)
        break