import pytest 
from .dataset import FMNIST, FMNISTConfig 
from module.config.env import get_env_config

@pytest.mark.current 
def test_fmnist_train():
    env_config = get_env_config()
    config = FMNISTConfig(env_config)
    dataset = FMNIST(config)
    
    assert len(dataset) == 60000
    assert dataset.batch_size == 64
    sample, label = dataset[0] 
    assert sample.shape == (1, 28, 28)
    assert label == 9
    for sample, label in dataset:
        assert sample.shape == (64, 1, 28, 28)
        assert label.shape == (64,)
        break
