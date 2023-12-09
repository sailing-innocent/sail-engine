import pytest 


from module.dataset.nvs.blender.dataset import NeRFBlenderDatasetConfig, NeRFBlenderDataset
from app.visualizer.multi_view.blender import MultiViewBlenderVisualizerConfig, MultiViewBlenderVisualizer
from mission.config.env import get_env_config

@pytest.mark.vis
def test_vis_nerf_blender():
    env_config = get_env_config()
    dataset_config = NeRFBlenderDatasetConfig(env_config)
    vis_config = MultiViewBlenderVisualizerConfig(env_config)
    for obj_name in dataset_config.obj_list:
        dataset_config.obj_name = obj_name
        vis_config.mainfile_name = obj_name
        dataset = NeRFBlenderDataset(dataset_config)
        visualizer = MultiViewBlenderVisualizer(vis_config)
        visualizer.visualize(dataset)
        break
    assert True

from module.dataset.nvs.mip360.dataset import Mip360DatasetConfig, Mip360Dataset
@pytest.mark.vis
def test_vis_nerf_mipmap():
    env_config = get_env_config()
    config = Mip360DatasetConfig(env_config)
    vis_config = MultiViewBlenderVisualizerConfig(env_config)
    for obj_name in config.obj_list:
        config.obj_name = obj_name
        dataset = Mip360Dataset(config)
        vis_config.mainfile_name = obj_name
        visualizer = MultiViewBlenderVisualizer(vis_config)
        visualizer.visualize(dataset)
        break

    assert True

from module.dataset.nvs.tank_temple.dataset import TankTempleDatasetConfig, TankTempleDataset
@pytest.mark.current 
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
        break
    assert True