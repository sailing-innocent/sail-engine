from module.data.point_cloud import sphere_point_cloud
from module.utils.camera.basic import Camera
from module.model.gaussian.vanilla import GaussianModel
import numpy as np 
from app.diff_renderer.gaussian_rasterizer.vanilla import create_gaussian_renderer as create_vanilla_renderer 
from app.diff_renderer.gaussian_rasterizer.orth import GaussianRenderer
from mission.config.env import get_env_config
import matplotlib.pyplot as plt

def test_orth():
    env_config = get_env_config()
    source_gs = GaussianModel(3)
    r = 1.0
    N = 1000
    red = [1, 0, 0]
    blue = [0, 0, 1]
    pcd = sphere_point_cloud(r, N, blue)
    source_gs.create_from_pcd(pcd, r)
    # source_gs.load_ply("D:/pretrained/gaussian/nerf_blender_lego_30000.ply")
    cam = Camera("FlipY")
    cam.lookat(2 * np.array([1, 0, 1]), np.array([0, 0, 0]))
    inno_split_renderer = GaussianRenderer(env_config, 1)
    
    target_img = inno_split_renderer.render(cam, source_gs)["render"]
    target_img = target_img.detach()
    target_img_np = target_img.cpu().numpy().transpose(1, 2, 0)
    target_img_np = target_img_np[::-1, :, :]
    target_img_np = target_img_np.clip(0, 1)

    plt.imshow(target_img_np)
    plt.show()