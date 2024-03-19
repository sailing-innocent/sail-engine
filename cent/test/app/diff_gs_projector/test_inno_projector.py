from module.data.point_cloud import sphere_point_cloud
from module.model.gaussian.vanilla import GaussianModel
from module.utils.camera.basic import Camera
from mission.config.env import get_env_config

from app.diff_renderer.gaussian_projector.inno import GaussianProjector

import numpy as np 
import pytest 

@pytest.mark.app
def test_diff_gs_projector():
    env_config = get_env_config()
    source_gs = GaussianModel(3)
    r = 1.0
    N = 5
    red = [1, 0, 0]
    blue = [0, 0, 1]
    pcd = sphere_point_cloud(r, N, blue)
    cam = Camera("FlipY")
    source_gs.create_from_pcd(pcd, r)
    cam.lookat(2 * np.array([0, -1, 0]), np.array([0, 0, 0]))

    print(source_gs.get_xyz)
    print(source_gs.get_features)
    print(source_gs.get_scaling)
    print(source_gs.get_rotation)
    print(source_gs.get_opacity)

    projector = GaussianProjector()
    gaussians_2d = projector.project(source_gs, cam)

    print(gaussians_2d.means_2d)
    print(gaussians_2d.covs_2d)
    print(gaussians_2d.depth_features)
    print(gaussians_2d.color_features)
