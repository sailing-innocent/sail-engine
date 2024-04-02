from module.model.gaussian.zzh import GaussianModel
from module.dataset.nvs.blender.dataset import create_dataset as create_nerf_blender_dataset
from mission.config.env import get_env_config

def test_split():
    gs = GaussianModel(3)
    env_config = get_env_config()
    dataset = create_nerf_blender_dataset(env_config, 'lego', 'train')
    pcd = dataset.get_point_cloud()
    gs.create_from_pcd(pcd, 1.0)
    gs.save_ply('D:/pretrained_random.ply')