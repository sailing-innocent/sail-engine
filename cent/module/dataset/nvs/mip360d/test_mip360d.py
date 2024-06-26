import pytest 
from .dataset import Mip360DatasetConfig, Mip360Dataset
from mission.config.env import get_env_config
import matplotlib.pyplot as plt

@pytest.mark.current
def test_mip360d():
    env_config = get_env_config()
    config = Mip360DatasetConfig(env_config)
    dataset = Mip360Dataset(config)
    assert len(dataset) == 194
    for cam, img in dataset:
        assert img.data.shape == (822, 1237, 4)
        color_img = img.data[:, :, :3]
        depth_img = img.data[:, :, 3]
        plt.imshow(color_img)
        plt.imshow(depth_img)
        plt.show()
        assert cam.R.shape == (3, 3)
        assert cam.T.shape == (3, )
        break

    assert True 
