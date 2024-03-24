import pytest 

from .dataset import AQIDataset, AQIDatasetConfig
from mission.config.env import get_env_config

@pytest.mark.current 
def test_aqi_dataset():
    env_config = get_env_config()
    dataset_config = AQIDatasetConfig(env_config)
    dataset = AQIDataset(dataset_config)
    # print(dataset_config.csv_file_path)
    assert len(dataset) == 323
    sample = dataset[0]
    assert len(sample) == 11
    assert dataset.batch_size == 10
    for feature, label in dataset:
        assert feature.shape == (10, 10)
        print(feature[0])
        assert label.shape == (10, )
        print(label[0])
        break