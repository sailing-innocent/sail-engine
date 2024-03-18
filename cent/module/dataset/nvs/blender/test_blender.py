import pytest 

from module.dataset.nvs.blender.dataset import NeRFBlenderDatasetConfig, NeRFBlenderDataset
from mission.config.env import get_env_config
import matplotlib.pyplot as plt

@pytest.mark.current 
def test_nerf_blender():
    env_config = get_env_config()
    config = NeRFBlenderDatasetConfig(env_config)
    dataset = NeRFBlenderDataset(config)
    assert len(dataset) == 100

    for cam, img in dataset:
        assert img.data.shape == (800, 800, 3)
        plt.imshow(img.data)
        plt.show()
        assert cam.R.shape == (3, 3)
        assert cam.T.shape == (3, )
        break