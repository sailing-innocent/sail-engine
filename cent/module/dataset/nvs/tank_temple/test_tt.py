import pytest 
from .dataset import TankTempleDatasetConfig, TankTempleDataset
from mission.config.env import get_env_config
import matplotlib.pyplot as plt

@pytest.mark.current
def test_mip360():
    env_config = get_env_config()
    config = TankTempleDatasetConfig(env_config)
    dataset = TankTempleDataset(config)
    assert len(dataset) == 258
    for cam, img in dataset:
        assert img.data.shape == (546, 982, 3)
        plt.imshow(img.data)
        plt.show()
        assert cam.R.shape == (3, 3)
        assert cam.T.shape == (3, )
        break

    assert True 
