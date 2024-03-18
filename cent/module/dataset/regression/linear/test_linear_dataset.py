import pytest 
from .dataset import LinearDataset, LinearDatasetConfig
from module.config.env import get_env_config_by_name  

@pytest.mark.current
def test_linear_dataset():
    env_config = get_env_config_by_name("pc")
    config = LinearDatasetConfig(env_config)
    dataset = LinearDataset(config)

    assert str(dataset) == "linear"
    assert len(dataset) == 1000
    assert config.batch_size == 10
    # dataset.visualize()
    for sample, label in dataset:
        assert sample.shape == (10,2)
        assert label.shape == (10,1)
        break

    for i in range(10):
        sample, label = dataset[i]
        assert sample.shape == (2,)
        assert label.shape == (1,)

    features = dataset.features()
    labels = dataset.labels()

    assert features.shape == (1000, 2)
    assert labels.shape == (1000, 1)
    
    dataset.visualize()