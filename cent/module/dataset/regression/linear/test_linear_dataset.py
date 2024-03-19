import pytest 
from .dataset import LinearDataset, LinearDatasetConfig
from mission.config.env import get_env_config

@pytest.mark.current
def test_linear_dataset():
    env_config = get_env_config()
    N = 100
    config = LinearDatasetConfig(env_config)
    config.sample_size = N
    dataset = LinearDataset(config)

    assert str(dataset) == "linear"
    assert len(dataset) == N
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

    assert features.shape == (N, 2)
    assert labels.shape == (N, 1)
    dataset.visualize()