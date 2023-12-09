import pytest 
from .dataset import MNIST, MNISTConfig 
from module.config.env import get_env_config


@pytest.mark.current 
def test_mnist():
    env_config = get_env_config()
    config = MNISTConfig(env_config)
    dataset = MNIST(config)
    
    assert len(dataset) == 60000
    assert dataset.batch_size == 64
    sample, label = dataset[0] 
    assert sample.shape == (1, 28, 28)
    assert label == 5
    for sample, label in dataset:
        assert sample.shape == (64, 1, 28, 28)
        assert label.shape == (64,)
        break
