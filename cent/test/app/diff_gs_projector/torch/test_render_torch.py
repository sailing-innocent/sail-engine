from module.data.point_cloud import sphere_point_cloud
from module.model.gaussian.vanilla import GaussianModel
from module.utils.camera.basic import Camera
from mission.config.env import get_env_config

# from app.diff_renderer.gaussian_projector.inno import GaussianProjector
from app.diff_renderer.gaussian_projector.torch import GaussianProjector
from app.diff_renderer.gaussian_sampler.inno import GaussianSampler

import numpy as np 
import pytest 
import matplotlib.pyplot as plt

@pytest.mark.current 
def test_diff_gs_projector():
    env_config = get_env_config()
    source_gs = GaussianModel(3)
    sampler = GaussianSampler()
    r = 1.0
    N = 50
    red = [1, 0, 0]
    blue = [0, 0, 1]
    pcd = sphere_point_cloud(r, N, blue)
    source_gs.create_from_pcd(pcd, r)

    cam = Camera("FlipY")
    cam.lookat(2 * np.array([0, -1, 0]), np.array([0, 0, 0]))
    cam.set_res(1920, 1080)

    projector = GaussianProjector()
    gs_2d = projector.project(source_gs, cam)
    # print(gs_2d.means_2d)
    # print(gs_2d.covs_2d)
    # print(gs_2d.depth_features)
    # print(gs_2d.color_features)
    
    target_img = sampler.sample(gs_2d, cam.info.ResW, cam.info.ResH, cam.info.FovY)
    target_img_np = target_img.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)[::-1, :, :]
    plt.imshow(target_img_np)
    plt.show()
