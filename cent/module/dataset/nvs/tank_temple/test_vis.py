import pytest

from app.visualizer.multi_view.blender import MultiViewBlenderVisualizerConfig, MultiViewBlenderVisualizer
from module.dataset.nvs.tank_temple.dataset import TankTempleDatasetConfig, TankTempleDataset
from module.config.env import get_env_config

@pytest.mark.vis
def test_vis_nerf_mipmap():
    env_config = get_env_config()
    config = TankTempleDatasetConfig(env_config)
    vis_config = MultiViewBlenderVisualizerConfig(env_config)
    for obj_name in config.obj_list:
        config.obj_name = obj_name
        dataset = TankTempleDataset(config)
        vis_config.mainfile_name = obj_name
        visualizer = MultiViewBlenderVisualizer(vis_config)
        visualizer.visualize(dataset)
    assert True